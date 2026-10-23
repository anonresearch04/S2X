import numpy as np
import torch
import torch.nn.functional as F

import utils.datasets as datasets_utils
from models.gate_model import GATE_ROWS


def train(model, frequency_table, rows, labels, learning_rate=0.001, epochs=10, batch_size=32, device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    num_rows = frequency_table.shape[1]

    with torch.no_grad():
        model.conv1.weight.copy_(torch.from_numpy(frequency_table.transpose(2, 0, 1))[:, None])
        model.conv1.bias.zero_()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rows_t, labels_t = torch.as_tensor(np.array(rows)), torch.as_tensor(np.array(labels))

    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(rows_t))
        epoch_loss = 0.0
        correct = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            yb = labels_t[idx].to(device)
            logits = model(datasets_utils.rows_to_onehot(rows_t[idx], num_rows, device))
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()

        print(f"Gate epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(order):.4f} | Accuracy: {correct / len(order) * 100:.2f}%")
    return model


def train_frequency_table(rows, labels, label_size, num_rows=GATE_ROWS):
    max_length = rows.shape[1]
    counts = np.zeros((max_length, num_rows, label_size), dtype=np.float64)
    positions = np.broadcast_to(np.arange(max_length), rows.shape)
    np.add.at(counts, (positions.ravel(), rows.ravel(), np.repeat(labels, max_length)), 1.0)
    totals = counts.sum(axis=2, keepdims=True)
    table = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return table.astype(np.float32)
