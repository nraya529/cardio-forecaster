"""Time-series Transformer for multivariate vital-sign forecasting.

Architecture summary
--------------------
- Channel projection + sinusoidal positional encoding
- Stack of encoder layers operating on the history window
- Causal decoder layers that autoregressively roll out the forecast horizon
- A final linear head projects back to the original channel space

The forward pass mirrors the encoder/decoder split used in
"Attention is All You Need" but with continuous-valued targets, so the loss
becomes regression (MSE/Huber) rather than cross-entropy. A small risk head
emits a per-step deterioration probability alongside the signal forecast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    n_channels: int = 6
    history: int = 360
    horizon: int = 120
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        # NOTE: registering as buffer (not attribute) so it follows .to(device).
        # persistent=False keeps it out of the state dict — it's deterministic, no need to save.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.n_channels, config.d_model)
        self.target_proj = nn.Linear(config.n_channels, config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        self.signal_head = nn.Linear(config.d_model, config.n_channels)
        self.risk_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def _causal_mask(self, size: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, history: Tensor) -> Tensor:
        h = self.pos_enc(self.input_proj(history))
        return self.encoder(h)

    def forward(self, history: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Teacher-forced training forward.

        Parameters
        ----------
        history : (B, history, C)
        target  : (B, horizon, C)  ground truth, shifted right inside this fn

        Returns
        -------
        signal_pred : (B, horizon, C)
        risk_logits : (B, horizon, 1)
        """
        memory = self.encode(history)
        sos = history[:, -1:, :]
        decoder_in = torch.cat([sos, target[:, :-1, :]], dim=1)
        decoder_emb = self.pos_enc(self.target_proj(decoder_in))
        mask = self._causal_mask(decoder_emb.size(1), decoder_emb.device)
        decoded = self.decoder(decoder_emb, memory, tgt_mask=mask)
        return self.signal_head(decoded), self.risk_head(decoded)

    @torch.inference_mode()
    def autoregressive_forecast(self, history: Tensor) -> tuple[Tensor, Tensor]:
        """Inference-time roll-out — no teacher forcing.

        Parameters
        ----------
        history : (B, history, C)

        Returns
        -------
        signal_pred : (B, horizon, C)
        risk_probs  : (B, horizon)
        """
        # TODO: this is the main inference cost — `horizon` forward passes.
        # Want to try parallel decoding (emit all H steps from learned queries
        # in one shot) once I can validate it on real data.
        memory = self.encode(history)
        current = history[:, -1:, :]
        signal_outputs: list[Tensor] = []
        risk_outputs: list[Tensor] = []

        for _ in range(self.config.horizon):
            decoder_emb = self.pos_enc(self.target_proj(current))
            mask = self._causal_mask(decoder_emb.size(1), decoder_emb.device)
            decoded = self.decoder(decoder_emb, memory, tgt_mask=mask)
            next_signal = self.signal_head(decoded[:, -1:, :])
            next_risk = torch.sigmoid(self.risk_head(decoded[:, -1:, :]))
            signal_outputs.append(next_signal)
            risk_outputs.append(next_risk)
            current = torch.cat([current, next_signal], dim=1)

        return (
            torch.cat(signal_outputs, dim=1),
            torch.cat(risk_outputs, dim=1).squeeze(-1),
        )


def build_model(config: TransformerConfig) -> TimeSeriesTransformer:
    return TimeSeriesTransformer(config)
