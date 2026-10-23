import numpy as np
import torch

from models.pl_bert_model import CLS_ID, SEP_ID, PAD_ID
from utils.preprocessing import gate_matrix, pl_bert_preprocessing


def get_pl_bert_tensors(df, max_length):
    tokens = np.array([pl_bert_preprocessing(x, max_length) for x in df["x"]], dtype=np.int64)
    ids = np.concatenate([np.full((len(tokens), 1), CLS_ID), tokens, np.full((len(tokens), 1), SEP_ID)], axis=1)
    mask = (ids != PAD_ID).astype(np.int64)
    return torch.from_numpy(ids), torch.from_numpy(mask), torch.tensor(df["y"].to_numpy(dtype=np.int64))


def get_gate_rows(df, max_length):
    return gate_matrix(df["x"], max_length), df["y"].to_numpy(dtype=np.int64)


def rows_to_onehot(rows, num_rows, device):
    batch, n = rows.shape
    out = torch.zeros(batch, 1, n, num_rows, device=device)
    out.scatter_(3, rows.to(device)[:, None, :, None], 1.0)
    return out
