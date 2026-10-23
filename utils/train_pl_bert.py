import numpy as np
import torch

import models.pl_bert_model as pl_bert_model
from utils.inference import pl_bert_predict_proba


class Stalled(RuntimeError):
    pass


def train(model, train_data, valid_data, device, epochs=10, batch_size=32, lr=5e-5, warmup_ratio=0.1,
          max_grad_norm=1.0, stall_check=True):
    (ids, mask, labels), (valid_ids, valid_mask, valid_y) = train_data, valid_data
    valid_y = valid_y.numpy()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps = epochs * -(-len(ids) // batch_size)
    warmup = int(warmup_ratio * steps)

    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(ids))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            out = model(input_ids=ids[idx].to(device), attention_mask=mask[idx].to(device), labels=labels[idx].to(device))
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
        proba = pl_bert_predict_proba(model, valid_ids, valid_mask, device)
        valid_loss = float(-np.log(np.clip(proba[np.arange(len(valid_y)), valid_y], 1e-12, None)).mean())
        valid_acc = float((proba.argmax(1) == valid_y).mean())
        print(f"PL-BERT epoch {epoch+1}/{epochs} | Valid loss: {valid_loss:.4f} | Valid accuracy: {valid_acc*100:.2f}%")
        chance = 2.0 / model.config.num_labels
        if stall_check and epoch in (0, epochs - 1) and valid_acc <= chance:
            raise Stalled(f"valid accuracy {valid_acc:.4f} <= {chance:.4f} after epoch {epoch+1}")
    return model


def fit(num_labels, train_data, valid_data, device, seed=42, retries=3, **hp):
    for s in range(seed, seed + retries):
        torch.manual_seed(s)
        model = pl_bert_model.get_model(num_labels)
        try:
            return train(model, train_data, valid_data, device, **hp), s
        except Stalled as err:
            print(f"PL-BERT seed {s} stalled ({err}); re-initialising with seed {s + 1}")
    raise RuntimeError(f"PL-BERT stalled for seeds {seed}..{seed + retries - 1}")
