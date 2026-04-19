"""MIMIC-III Waveform Database loader.

The actual files require credentialed access from PhysioNet. This loader stays
schema-compatible with `synthetic.generate_episode` so swapping data sources
needs no code changes downstream.

Usage
-----
loader = MimicWaveformLoader(root="data/mimic3wdb")
for episode_id, signals, labels in loader.iter_episodes():
    ...

The on-disk layout we expect is::

    data/mimic3wdb/
        <subject_id>/
            <record_id>.hea
            <record_id>.dat
            <record_id>.labels.npy  # optional, downstream of deterioration labelling
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CHANNELS = ("hr", "sbp", "dbp", "spo2", "resp", "temp")


class MimicWaveformLoader:
    """Stream multi-channel vital-sign records from the MIMIC-III waveform set."""

    def __init__(
        self,
        root: str | Path,
        channels: tuple[str, ...] = CHANNELS,
        target_hz: float = 1.0,
    ) -> None:
        self.root = Path(root)
        self.channels = channels
        self.target_hz = target_hz
        if not self.root.exists():
            logger.warning(
                "MIMIC root %s does not exist. Falling back to synthetic data is the caller's responsibility.",
                self.root,
            )

    def iter_episodes(self) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        if not self.root.exists():
            return
        try:
            import wfdb  # noqa: WPS433 - optional dependency
        except ImportError as err:  # pragma: no cover - documented optional path
            raise ImportError(
                "wfdb is required to read MIMIC waveform records. "
                "Install with `pip install wfdb`."
            ) from err

        for header in sorted(self.root.rglob("*.hea")):
            record_name = str(header.with_suffix("")).removeprefix(str(self.root) + "/")
            try:
                record = wfdb.rdrecord(str(header.with_suffix("")))
            except Exception as err:  # noqa: BLE001 - skip corrupt records
                logger.warning("Skipping record %s: %s", record_name, err)
                continue

            signal_map = {name.lower(): idx for idx, name in enumerate(record.sig_name)}
            if not all(ch in signal_map for ch in self.channels):
                logger.debug("Record %s missing channels, skipping.", record_name)
                continue

            cols = [signal_map[ch] for ch in self.channels]
            signals = record.p_signal[:, cols].astype(np.float32)
            signals = self._resample(signals, record.fs, self.target_hz)

            labels_path = header.with_name(header.stem + ".labels.npy")
            labels = (
                np.load(labels_path).astype(np.uint8)
                if labels_path.exists()
                else np.zeros(signals.shape[0], dtype=np.uint8)
            )
            yield record_name, signals, labels

    @staticmethod
    def _resample(signals: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
        if abs(src_hz - dst_hz) < 1e-6:
            return signals
        step = src_hz / dst_hz
        if step <= 0:
            raise ValueError(f"Invalid resample ratio: src={src_hz} dst={dst_hz}")
        n_out = int(signals.shape[0] / step)
        idx = (np.arange(n_out) * step).astype(np.int64)
        return signals[idx]
