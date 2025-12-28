import torch
import torch.nn as nn


class MultiTaskLSTM(nn.Module):
    """LSTM-based multi-task model.

    Outputs a dict with keys:
      - 'quantiles': Tensor (B, num_quantiles)
      - 'vol': Tensor (B,)  # predicted volatility
      - 'trend_logits': Tensor (B, num_trend_classes)
      - 'cvar': Tensor (B,)  # conditional VaR estimate
      - 'risk_logits': Tensor (B, num_risk_classes)

    This module focuses on sequence encoding (LSTM) and lightweight heads for each task.
    """

    def __init__(self, in_dim, hidden=128, layers=2, num_quantiles=5, dropout=0.2,
                 num_trend_classes=2, num_risk_classes=3):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.hidden = hidden

        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False
        )

        # shared projection
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # heads
        self.quantile_head = nn.Linear(hidden, num_quantiles)
        self.vol_head = nn.Linear(hidden, 1)
        self.cvar_head = nn.Linear(hidden, 1)
        self.trend_head = nn.Linear(hidden, num_trend_classes)
        self.risk_head = nn.Linear(hidden, num_risk_classes)

    def forward(self, x):
        # x: (B, T, F)
        h, _ = self.rnn(x)  # (B, T, H)
        last = h[:, -1, :]
        z = self.proj(last)

        quantiles = self.quantile_head(z)
        vol = self.vol_head(z).squeeze(-1)
        cvar = self.cvar_head(z).squeeze(-1)
        trend_logits = self.trend_head(z)
        risk_logits = self.risk_head(z)

        return {
            'quantiles': quantiles,
            'vol': vol,
            'cvar': cvar,
            'trend_logits': trend_logits,
            'risk_logits': risk_logits
        }
