"""Bi-LSTM baseline.

Kept around as a reference point: a much smaller model that the Transformer
should outperform on long-horizon forecasts. Useful for ablation and for
local debugging on machines without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class LSTMConfig:
    n_channels: int = 6
    horizon: int = 120
    hidden: int = 128
    n_layers: int = 2
    dropout: float = 0.1


class BiLSTMForecaster(nn.Module):
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            input_size=config.n_channels,
            hidden_size=config.hidden,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.bridge = nn.Linear(2 * config.hidden, config.hidden)
        self.decoder = nn.LSTM(
            input_size=config.n_channels,
            hidden_size=config.hidden,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.signal_head = nn.Linear(config.hidden, config.n_channels)
        self.risk_head = nn.Linear(config.hidden, 1)

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        _, (h, c) = self.encoder(history)
        h = h.view(self.config.n_layers, 2, -1, self.config.hidden).mean(dim=1)
        c = c.view(self.config.n_layers, 2, -1, self.config.hidden).mean(dim=1)
        sos = history[:, -1:, :]
        signal_outs: list[Tensor] = []
        risk_outs: list[Tensor] = []
        current = sos
        state = (h.contiguous(), c.contiguous())
        for _ in range(self.config.horizon):
            out, state = self.decoder(current, state)
            signal = self.signal_head(out)
            risk = torch.sigmoid(self.risk_head(out)).squeeze(-1)
            signal_outs.append(signal)
            risk_outs.append(risk)
            current = signal
        return torch.cat(signal_outs, dim=1), torch.cat(risk_outs, dim=1)
