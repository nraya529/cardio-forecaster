"""Evaluate a checkpoint on freshly generated episodes."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data.synthetic import generate_episode  # noqa: E402
from app.inference.predictor import CardioForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def evaluate(checkpoint: Path, n_episodes: int, seed: int) -> dict:
    predictor = CardioForecaster()
    predictor.load(checkpoint)
    history = predictor.settings.history_window
    horizon = predictor.settings.forecast_horizon
    needed = history + horizon + 1
    minutes = max(needed // 60 + 5, 90)

    rng = np.random.default_rng(seed)
    mae_per_channel = np.zeros(predictor.settings.n_channels)
    risk_correct = 0
    for _ in range(n_episodes):
        signals, labels, _ = generate_episode(
            n_minutes=minutes,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        history_chunk = signals[-(history + horizon): -horizon]
        target = signals[-horizon:]
        target_risk = labels[-horizon:]
        forecast = predictor.forecast(history_chunk)
        mae_per_channel += np.mean(np.abs(forecast.signals - target), axis=0)
        risk_correct += int((forecast.risk.max() > 0.5) == bool(target_risk.any()))

    mae_per_channel /= n_episodes
    risk_acc = risk_correct / n_episodes
    return {
        "mae_per_channel": dict(zip(predictor.settings.channel_names, mae_per_channel.tolist(), strict=True)),
        "deterioration_detection_acc": risk_acc,
        "n_episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/transformer_best.pt"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, args.episodes, args.seed)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
