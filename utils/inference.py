import numpy as np
import torch

from models.pl_bert_model import CLS_ID, SEP_ID, PAD_ID
from utils.preprocessing import GATE_OFFSET, TOKEN_OFFSET


def pl_bert_predict_proba(model, ids, mask, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = slice(start, start + batch_size)
            logits = model(input_ids=ids[batch].to(device), attention_mask=mask[batch].to(device)).logits
            out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out).numpy()


def gate_inference(table, rows, chunk=5000):
    preds, scores = [], []
    positions = np.arange(rows.shape[1])
    for start in range(0, len(rows), chunk):
        sums = table[positions, rows[start:start + chunk]].sum(axis=1)
        proba = torch.softmax(torch.from_numpy(sums), dim=1).numpy()
        pred = proba.argmax(axis=1)
        preds.append(pred)
        scores.append(proba[np.arange(len(pred)), pred])
    return np.concatenate(preds), np.concatenate(scores)



def gate_inference_one(table, row):
    sums = table[np.arange(len(row)), row].sum(axis=0)
    proba = torch.softmax(torch.from_numpy(sums), dim=0)
    pred = int(proba.argmax())
    return pred, float(proba[pred])


def pl_bert_inference_one(model, prefix, device):
    tokens = prefix + TOKEN_OFFSET
    ids = torch.from_numpy(np.concatenate([[CLS_ID], tokens, [SEP_ID]]))[None].to(device)
    with torch.no_grad():
        proba = torch.softmax(model(input_ids=ids, attention_mask=(ids != PAD_ID).long()).logits, dim=-1)[0]
    pred = int(proba.argmax())
    return pred, float(proba[pred])


def gate_rows_one(prefix):
    return np.where(prefix == -TOKEN_OFFSET, 0, prefix) + GATE_OFFSET
