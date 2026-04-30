"""Train the time-series Transformer on synthetic or MIMIC-III episodes.

Usage
-----
python training/train.py --config training/config.yaml

The CLI prefers a YAML config so experiments are reproducible. Anything passed
on the command line overrides the YAML.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data.preprocessing import ChannelStats  # noqa: E402
from app.data.synthetic import generate_episode, streaming_window  # noqa: E402
from app.inference.predictor import CardioForecaster  # noqa: E402
from app.models.transformer import TimeSeriesTransformer, TransformerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    val_split: float = 0.15
    n_episodes: int = 200
    episode_minutes: int = 720
    stride: int = 60
    risk_loss_weight: float = 0.2
    grad_clip: float = 1.0
    device: str = "auto"
    output_dir: Path = Path("artifacts")
    seed: int = 1337
    model: dict = field(default_factory=dict)


class WindowDataset(Dataset):
    def __init__(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        risk_labels: np.ndarray,
    ) -> None:
        assert len(x_norm) == len(y_norm) == len(risk_labels)
        self.x = torch.from_numpy(x_norm).float()
        self.y = torch.from_numpy(y_norm).float()
        self.risk = torch.from_numpy(risk_labels).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.risk[idx]


def build_dataset(cfg: TrainConfig, model_cfg: TransformerConfig) -> tuple[WindowDataset, ChannelStats]:
    rng = np.random.default_rng(cfg.seed)
    raw_episodes: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    risk_chunks: list[np.ndarray] = []

    needed = model_cfg.history + model_cfg.horizon
    n_seconds = max(cfg.episode_minutes * 60, needed + cfg.stride)
    cfg_minutes = math.ceil(n_seconds / 60)

    for _ in range(cfg.n_episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        signals, labels, _ = generate_episode(n_minutes=cfg_minutes, seed=seed)
        raw_episodes.append(signals)
        x, y = streaming_window(signals, history=model_cfg.history, horizon=model_cfg.horizon, stride=cfg.stride)
        risk = np.zeros((y.shape[0], y.shape[1]), dtype=np.float32)
        for w_idx in range(y.shape[0]):
            start = w_idx * cfg.stride + model_cfg.history
            risk[w_idx] = labels[start:start + model_cfg.horizon]
        x_chunks.append(x)
        y_chunks.append(y)
        risk_chunks.append(risk)

    x_all = np.concatenate(x_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    risk_all = np.concatenate(risk_chunks, axis=0)
    stats = ChannelStats.from_episodes(raw_episodes)
    x_norm = stats.normalize(x_all.reshape(-1, model_cfg.n_channels)).reshape(x_all.shape)
    y_norm = stats.normalize(y_all.reshape(-1, model_cfg.n_channels)).reshape(y_all.shape)
    return WindowDataset(x_norm.astype(np.float32), y_norm.astype(np.float32), risk_all), stats


def lr_lambda(step: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.05, 1.0 - (step - warmup) / 10_000)


def train(cfg: TrainConfig, model_cfg: TransformerConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
    )
    logger.info("Building dataset (%d episodes, %d min each)...", cfg.n_episodes, cfg.episode_minutes)
    dataset, stats = build_dataset(cfg, model_cfg)
    val_size = int(len(dataset) * cfg.val_split)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    logger.info("Dataset: %d train windows | %d val windows", len(train_set), len(val_set))

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    model = TimeSeriesTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg.warmup_steps))
    signal_loss = nn.SmoothL1Loss()
    risk_loss = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.output_dir / "transformer_best.pt"
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        for x, y, risk in train_loader:
            x, y, risk = x.to(device), y.to(device), risk.to(device)
            optimizer.zero_grad()
            pred, risk_logits = model(x, y)
            sig = signal_loss(pred, y)
            rsk = risk_loss(risk_logits.squeeze(-1), risk)
            loss = sig + cfg.risk_loss_weight * rsk
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

        val_loss = _evaluate(model, val_loader, device, signal_loss, risk_loss, cfg.risk_loss_weight)
        logger.info(
            "epoch=%d val_loss=%.4f best=%.4f elapsed=%.1fs",
            epoch, val_loss, best_val, time.time() - epoch_start,
        )
        if val_loss < best_val:
            best_val = val_loss
            predictor = CardioForecaster()
            predictor.model = model
            predictor.stats = stats
            predictor.save(best_path, extra={"epoch": epoch, "val_loss": val_loss})
            logger.info("Saved new best to %s", best_path)

    return best_path


def _evaluate(model, loader, device, signal_loss, risk_loss, w: float) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for x, y, risk in loader:
            x, y, risk = x.to(device), y.to(device), risk.to(device)
            pred, logits = model(x, y)
            losses.append((signal_loss(pred, y) + w * risk_loss(logits.squeeze(-1), risk)).item())
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    train_cfg = TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig.__dataclass_fields__})
    model_cfg = TransformerConfig(**raw.get("model", {}))
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.device is not None:
        train_cfg.device = args.device
    logger.info("Train config: %s", asdict(train_cfg))
    logger.info("Model config: %s", asdict(model_cfg))
    out = train(train_cfg, model_cfg)
    logger.info("Done. Best checkpoint: %s", out)


if __name__ == "__main__":
    main()
