"""Model loading and forecast post-processing."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.core.config import Settings, get_settings
from app.data.preprocessing import ChannelStats
from app.models.transformer import TimeSeriesTransformer, TransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class Forecast:
    signals: np.ndarray
    risk: np.ndarray
    risk_peak: float
    risk_peak_minute: int


class CardioForecaster:
    """Wraps the Torch model with normalization, batching, and risk smoothing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TimeSeriesTransformer | None = None
        self.stats: ChannelStats | None = None

    def load(self, model_path: Path | None = None) -> None:
        path = Path(model_path or self.settings.model_path)
        if not path.exists():
            logger.warning("Model checkpoint %s not found — initializing untrained model.", path)
            self._init_untrained()
            return
        checkpoint = torch.load(path, map_location=self.device)
        config = TransformerConfig(**checkpoint["config"])
        self.model = TimeSeriesTransformer(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.stats = ChannelStats.from_dict(checkpoint["stats"])
        logger.info("Loaded checkpoint from %s (device=%s)", path, self.device)

    def _init_untrained(self) -> None:
        config = TransformerConfig(
            n_channels=self.settings.n_channels,
            history=self.settings.history_window,
            horizon=self.settings.forecast_horizon,
            d_model=self.settings.d_model,
            n_heads=self.settings.n_heads,
            n_encoder_layers=self.settings.n_encoder_layers,
            n_decoder_layers=self.settings.n_decoder_layers,
            dim_feedforward=self.settings.dim_feedforward,
            dropout=self.settings.dropout,
        )
        self.model = TimeSeriesTransformer(config).to(self.device).eval()
        # Identity stats so denormalize is a no-op.
        self.stats = ChannelStats(
            mean=np.zeros(self.settings.n_channels, dtype=np.float32),
            std=np.ones(self.settings.n_channels, dtype=np.float32),
        )

    def forecast(self, history: np.ndarray) -> Forecast:
        if self.model is None or self.stats is None:
            raise RuntimeError("Predictor not initialized. Call load() first.")
        if history.shape[0] != self.settings.history_window:
            raise ValueError(
                f"Expected history of length {self.settings.history_window}, got {history.shape[0]}."
            )
        if history.shape[1] != self.settings.n_channels:
            raise ValueError(
                f"Expected {self.settings.n_channels} channels, got {history.shape[1]}."
            )
        normalized = self.stats.normalize(history)
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)
        signals, risk = self.model.autoregressive_forecast(tensor)
        signals_np = self.stats.denormalize(signals.squeeze(0).cpu().numpy())
        risk_np = risk.squeeze(0).cpu().numpy()
        smoothed = _smooth(risk_np, window=5)
        peak_minute = int(smoothed.argmax())
        return Forecast(
            signals=signals_np,
            risk=smoothed,
            risk_peak=float(smoothed.max()),
            risk_peak_minute=peak_minute,
        )

    def save(self, path: Path, extra: dict | None = None) -> None:
        if self.model is None or self.stats is None:
            raise RuntimeError("Cannot save an uninitialized model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "config": self.model.config.__dict__,
            "stats": self.stats.to_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)
        path.with_suffix(".meta.json").write_text(json.dumps(payload["config"], indent=2))


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")
