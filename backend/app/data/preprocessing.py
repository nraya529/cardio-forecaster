"""Channel-wise normalization and reconstruction.

The model trains on z-scored signals, but the API returns predictions in
clinical units so the dashboard can plot them next to raw observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_episodes(cls, episodes: list[np.ndarray]) -> ChannelStats:
        stacked = np.concatenate(episodes, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> ChannelStats:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )
