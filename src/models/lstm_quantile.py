import torch
import torch.nn as nn

class LSTMQuantile(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=2, num_quantiles=5, dropout=0.2):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, num_quantiles)

    def forward(self, x):
        h, _ = self.rnn(x)          # (B, T, H)
        out = self.fc(h[:, -1, :])  # (B, num_quantiles)
        return out