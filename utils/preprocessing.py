import os
import json
import numpy as np
import pandas as pd

MTU = 1500
GATE_OFFSET = MTU
TOKEN_OFFSET = MTU + 1000
UNKNOWN = -1
HOLDOUT_SEED_BASE = 20260920
HOLDOUT_STRATA = 4



def encode_session(lengths):
    lengths = np.clip(np.asarray(lengths, dtype=np.int64), -MTU, MTU)
    payload = np.flatnonzero(lengths)
    return lengths[payload[0]:] if len(payload) else lengths[:0]


def convert_flow_fix_size(seq, max_length, padding):
    seq = np.asarray(seq, dtype=np.int64)[:max_length]
    return seq.tolist() + [padding] * (max_length - len(seq))


def gate_preprocessing(x, max_length):
    return [v + GATE_OFFSET for v in convert_flow_fix_size(x, max_length, 0)]


def pl_bert_preprocessing(x, max_length):
    return [v + TOKEN_OFFSET for v in convert_flow_fix_size(x, max_length, -TOKEN_OFFSET)]


def gate_matrix(xs, max_length):
    return np.array([gate_preprocessing(x, max_length) for x in xs], dtype=np.int64)



def get_label_map(train_df_path, y_label):
    labels = pd.read_parquet(train_df_path, columns=[y_label])[y_label]
    return {label: idx for idx, label in enumerate(sorted(labels.unique()))}


def convert_to_df(df, x_key, y_key, label_map):
    xs = [encode_session(x) for x in df[x_key]]
    ys = df[y_key].map(label_map)
    if ys.isna().any():
        raise ValueError(f"{int(ys.isna().sum())} sessions have a label that is not in the training split")
    return pd.DataFrame({"id": np.arange(len(df)), "x": xs, "y": ys.to_numpy(dtype=np.int64)})


def get_preprocessed_df(df_path, x_label, y_label, label_map, dataset_path):
    df = pd.read_parquet(df_path, columns=[x_label, y_label])
    origin_filename = os.path.basename(df_path).split(".")[0]
    preprocessed_df = convert_to_df(df, x_label, y_label, label_map)
    os.makedirs(dataset_path, exist_ok=True)
    preprocessed_df.to_parquet(os.path.join(dataset_path, f"preprocessed_{origin_filename}.parquet"))
    return preprocessed_df



def service_permutation(train_labels, num_classes, seed, strata=HOLDOUT_STRATA):
    counts = np.bincount(train_labels, minlength=num_classes)
    rng = np.random.default_rng([HOLDOUT_SEED_BASE, seed])
    order = np.argsort(counts, kind="stable")
    rank = np.empty(len(counts), dtype=np.float64)
    for part in np.array_split(order, min(strata, len(order))):
        shuffled = rng.permutation(part)
        rank[shuffled] = (np.arange(len(shuffled)) + rng.random(len(shuffled))) / len(shuffled)
    return np.argsort(rank + rng.random(len(rank)) * 1e-9, kind="stable")


def num_unknown(num_classes, ratio):
    return int(np.clip(round(num_classes * ratio / 100), 1, num_classes - 2))


def make_holdout(train_df, valid_df, test_df, num_classes, ratio, seed):
    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    order = service_permutation(train_df["y"].to_numpy(), num_classes, seed)
    unknown = np.sort(order[:num_unknown(num_classes, ratio)])
    known = np.sort(np.setdiff1d(np.arange(num_classes), unknown))
    is_unknown = np.zeros(num_classes, bool)
    is_unknown[unknown] = True
    relabel = np.full(num_classes, UNKNOWN, dtype=np.int64)
    relabel[known] = np.arange(len(known))

    kept, pool = {}, []
    for part, df in splits.items():
        mask = is_unknown[df["y"].to_numpy()]
        kept[part] = np.flatnonzero(~mask)
        pool += [(part, row) for row in np.flatnonzero(mask)]

    rng = np.random.default_rng([HOLDOUT_SEED_BASE, seed, ratio])
    picked = rng.choice(len(pool), size=min(len(kept["test"]), len(pool)), replace=False)
    unknown_rows = {part: [] for part in splits}
    for index in sorted(picked):
        part, row = pool[index]
        unknown_rows[part].append(row)

    def known_part(part):
        df = splits[part].iloc[kept[part]].copy()
        df["y"] = relabel[df["y"].to_numpy()]
        return df

    out = {part: known_part(part) for part in splits}
    unknown_df = pd.concat([splits[part].iloc[rows] for part, rows in unknown_rows.items() if rows])
    unknown_df = unknown_df.assign(y=UNKNOWN)
    open_test = pd.concat([out["test"], unknown_df])
    for df in (out["train"], out["valid"], open_test):
        df["id"] = np.arange(len(df))
    meta = {"ratio": ratio, "seed": seed, "num_known": int(len(known)),
            "known_classes": known.tolist(), "unknown_classes": unknown.tolist(),
            "train": len(out["train"]), "valid": len(out["valid"]),
            "test_known": len(out["test"]), "test_unknown": len(unknown_df)}
    return out["train"].reset_index(drop=True), out["valid"].reset_index(drop=True), open_test.reset_index(drop=True), meta


def get_holdout_dfs(train_df_path, valid_df_path, test_df_path, num_classes, ratio, seed, dataset_path):
    train_df, valid_df, test_df = (pd.read_parquet(p) for p in (train_df_path, valid_df_path, test_df_path))
    train_df, valid_df, open_test_df, meta = make_holdout(train_df, valid_df, test_df, num_classes, ratio, seed)
    out_dir = os.path.join(dataset_path, f"holdout_r{ratio}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(out_dir, "preprocessed_train_df.parquet"))
    valid_df.to_parquet(os.path.join(out_dir, "preprocessed_valid_df.parquet"))
    open_test_df.to_parquet(os.path.join(out_dir, "preprocessed_openworld_test_df.parquet"))
    with open(os.path.join(out_dir, "holdout.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir, meta
