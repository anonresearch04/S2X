import torch
import torch.nn as nn
import numpy as np

GATE_ROWS = 3001


class GateModel(nn.Module):

    def __init__(self, max_length, label_size, num_rows=GATE_ROWS):
        super(GateModel, self).__init__()
        torch.manual_seed(2025)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=label_size,
            kernel_size=(max_length, num_rows),
            stride=1,
            padding=0
        )
        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        return x

    def get_table(self):
        table = self.conv1.weight
        table = table.detach().cpu().numpy()
        table = np.squeeze(table, axis=1)
        table = np.transpose(table, (1, 2, 0))
        table = np.ascontiguousarray(table)
        return table
