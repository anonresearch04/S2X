import os
import sys
import json
import dotenv
import pandas as pd
import models.s2x as s2x_model
import utils.preprocessing as preprocessing_utils


dotenv.load_dotenv()


def env(key, default=None):
    value = os.getenv(key, default)
    return value.strip().strip('"') if isinstance(value, str) else value


def flag(key, default="False"):
    return env(key, default).lower() == "true"


def number(key, cast, default=None):
    value = env(key, default)
    return cast(value) if value not in (None, "") else None


args = {
    "DIM": number("DIM", int, "30"),
    "LABEL": number("LABEL", int),
    "OPENWORLD": flag("OPENWORLD"),
    "IS_TRAIN": flag("IS_TRAIN"),
    "MODE": env("MODE", "s2x-c"),
    "TRAIN_DF": env("TRAIN_DF"),
    "VALID_DF": env("VALID_DF"),
    "TEST_DF": env("TEST_DF"),
    "OPEN_TEST_DF": env("OPEN_TEST_DF"),
    "RAW_TRAIN_DF": env("RAW_TRAIN_DF"),
    "RAW_VALID_DF": env("RAW_VALID_DF"),
    "RAW_TEST_DF": env("RAW_TEST_DF"),
    "RAW_X_LABEL": env("RAW_X_LABEL", "SPLT-Data"),
    "RAW_Y_LABEL": env("RAW_Y_LABEL", "Label"),
    "DATASET_PATH": env("DATASET_PATH"),
    "HOLDOUT_RATIO": number("HOLDOUT_RATIO", int),
    "HOLDOUT_SEED": number("HOLDOUT_SEED", int, "0"),
    "SEED": number("SEED", int, "42"),
    "GATE_BATCH_SIZE": number("GATE_BATCH_SIZE", int, "32"),
    "GATE_LR": number("GATE_LR", float, "0.001"),
    "GATE_EPOCH": number("GATE_EPOCH", int, "10"),
    "PL_BERT_BATCH_SIZE": number("PL_BERT_BATCH_SIZE", int, "32"),
    "PL_BERT_EPOCH": number("PL_BERT_EPOCH", int, "10"),
    "PL_BERT_LR": number("PL_BERT_LR", float, "0.00005"),
    "PL_BERT_WARMUP_RATIO": number("PL_BERT_WARMUP_RATIO", float, "0.1"),
    "THETA0_PERCENTILE": number("THETA0_PERCENTILE", float, "5"),
    "THETA1_PERCENTILE": number("THETA1_PERCENTILE", float, "65"),
    "THETA2": number("THETA2", float, "0.97"),
    "GATE_TABLE": env("GATE_TABLE", "learned"),
    "MEASURE_THROUGHPUT": flag("MEASURE_THROUGHPUT"),
    "OUTPUT_PATH": env("OUTPUT_PATH", "./outputs"),
    "GATE_MODEL_PATH": env("GATE_MODEL_PATH"),
    "BERT_MODEL_PATH": env("BERT_MODEL_PATH"),
    "DEVICE": env("DEVICE", "cuda"),
}


def preprocess():
    label_map = preprocessing_utils.get_label_map(args["RAW_TRAIN_DF"], args["RAW_Y_LABEL"])
    paths = []
    for key in ("RAW_TRAIN_DF", "RAW_VALID_DF", "RAW_TEST_DF"):
        preprocessing_utils.get_preprocessed_df(args[key], args["RAW_X_LABEL"], args["RAW_Y_LABEL"], label_map,
                                                args["DATASET_PATH"])
        paths.append(os.path.join(args["DATASET_PATH"], f"preprocessed_{os.path.basename(args[key]).split('.')[0]}.parquet"))
    with open(os.path.join(args["DATASET_PATH"], "label_map.json"), "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"{len(label_map)} services -> {args['DATASET_PATH']}")
    if args["HOLDOUT_RATIO"] is not None:
        holdout(paths, len(label_map))


def holdout(paths=None, num_classes=None):
    paths = paths or [args["TRAIN_DF"], args["VALID_DF"], args["TEST_DF"]]
    num_classes = num_classes or int(pd.read_parquet(paths[0], columns=["y"])["y"].max()) + 1
    out_dir, meta = preprocessing_utils.get_holdout_dfs(*paths, num_classes, args["HOLDOUT_RATIO"],
                                                       args["HOLDOUT_SEED"], args["DATASET_PATH"])
    print(f"holdout r={meta['ratio']}% seed={meta['seed']}: {meta['num_known']} known services "
          f"(set LABEL={meta['num_known']}), {len(meta['unknown_classes'])} unseen -> {out_dir}")


def main():
    if args["MODE"] not in ("s2x-c", "s2x-g", "s2x-b"):
        raise ValueError(f"MODE must be s2x-c, s2x-g or s2x-b, not {args['MODE']}")
    s2x = s2x_model.S2X(args, mode=args["MODE"], openworld=args["OPENWORLD"])
    if args["IS_TRAIN"]:
        s2x.train_gate()
        s2x.train_pl_bert()
        print(f"models saved to {s2x.output_path}")
    else:
        s2x.load_model()
        s2x.setting_gate_thresholds()
        s2x.forward()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preprocess":
        preprocess()
    elif len(sys.argv) > 1 and sys.argv[1] == "holdout":
        holdout()
    else:
        main()
