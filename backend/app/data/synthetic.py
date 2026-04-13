"""Synthetic ICU vital-sign generator.

Real waveforms in the MIMIC-III Waveform Database require credentialed access.
This module produces physiologically-plausible multivariate signals that match
the schema expected by `MimicWaveformLoader`, so the training and inference
pipelines stay identical regardless of data source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatientPhenotype:
    """Per-patient bias terms applied on top of the population baselines."""

    age: float
    baseline_hr: float
    baseline_sbp: float
    baseline_dbp: float
    baseline_spo2: float
    baseline_resp: float
    baseline_temp: float
    deterioration_prob: float


CHANNELS = ("hr", "sbp", "dbp", "spo2", "resp", "temp")


def sample_patient(rng: np.random.Generator) -> PatientPhenotype:
    age = float(rng.normal(64, 14).clip(18, 95))
    return PatientPhenotype(
        age=age,
        baseline_hr=float(rng.normal(78, 9)),
        baseline_sbp=float(rng.normal(122, 12)),
        baseline_dbp=float(rng.normal(74, 8)),
        baseline_spo2=float(rng.normal(97, 1.2).clip(90, 100)),
        baseline_resp=float(rng.normal(16, 2)),
        baseline_temp=float(rng.normal(36.9, 0.3)),
        deterioration_prob=float(rng.beta(1.2, 6.0)),
    )


def _circadian(t: np.ndarray, period_min: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * (t / period_min) + phase)


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0, sigma, size=n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + eps[i]
    return out


def generate_episode(
    n_minutes: int = 720,
    sampling_rate_hz: float = 1.0,
    deterioration: bool | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, PatientPhenotype]:
    """Generate a multi-channel vital-sign episode.

    The deterioration pattern is a rough proxy for sepsis-onset: HR up, BP
    down, SpO2 down, RR up, temp up — ramped over 45-90 min. Not authoritative,
    but enough structure for the risk head to learn from. Real labels would
    come from a Sepsis-3 / MEWS / SOFA crossing on actual MIMIC data.

    Returns
    -------
    signals : (T, C) float32 array of vital signs, channel order matches `CHANNELS`.
    labels  : (T,) uint8 array, 1 inside a deterioration window, else 0.
    """
    rng = np.random.default_rng(seed)
    patient = sample_patient(rng)
    n = int(n_minutes * 60 * sampling_rate_hz)
    t = np.arange(n) / (60 * sampling_rate_hz)

    hr = patient.baseline_hr + _circadian(t, 90, 4.5) + _ar1(n, 0.95, 1.1, rng)
    sbp = patient.baseline_sbp + _circadian(t, 120, 6.5, phase=0.4) + _ar1(n, 0.97, 1.6, rng)
    dbp = patient.baseline_dbp + _circadian(t, 120, 3.0, phase=0.4) + _ar1(n, 0.97, 1.2, rng)
    spo2 = patient.baseline_spo2 + _ar1(n, 0.92, 0.25, rng)
    resp = patient.baseline_resp + _circadian(t, 70, 1.2) + _ar1(n, 0.9, 0.6, rng)
    temp = patient.baseline_temp + _circadian(t, 720, 0.35) + _ar1(n, 0.99, 0.05, rng)

    labels = np.zeros(n, dtype=np.uint8)
    will_deteriorate = (
        bool(rng.random() < patient.deterioration_prob) if deterioration is None else deterioration
    )
    if will_deteriorate:
        onset = int(rng.integers(n // 4, max(n // 4 + 1, int(n * 0.85))))
        window = min(n - onset, int(rng.integers(45 * 60, 90 * 60)))
        ramp = np.linspace(0, 1, window) ** 1.4
        hr[onset:onset + window] += 25 * ramp + rng.normal(0, 1.5, window)
        sbp[onset:onset + window] -= 22 * ramp + rng.normal(0, 1.8, window)
        dbp[onset:onset + window] -= 12 * ramp + rng.normal(0, 1.2, window)
        spo2[onset:onset + window] -= 4.5 * ramp + rng.normal(0, 0.3, window)
        resp[onset:onset + window] += 6 * ramp + rng.normal(0, 0.5, window)
        temp[onset:onset + window] += 0.9 * ramp + rng.normal(0, 0.05, window)
        labels[onset:onset + window] = 1

    spo2 = np.clip(spo2, 70.0, 100.0)
    temp = np.clip(temp, 34.0, 41.5)
    hr = np.clip(hr, 30.0, 220.0)

    signals = np.stack([hr, sbp, dbp, spo2, resp, temp], axis=1).astype(np.float32)
    return signals, labels, patient


def streaming_window(
    signals: np.ndarray,
    history: int,
    horizon: int,
    stride: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a (T, C) episode into overlapping (history, C) and (horizon, C) windows."""
    n = signals.shape[0]
    samples_x, samples_y = [], []
    for start in range(0, n - history - horizon + 1, stride):
        samples_x.append(signals[start:start + history])
        samples_y.append(signals[start + history:start + history + horizon])
    if not samples_x:
        raise ValueError(f"Episode too short for window: n={n} history={history} horizon={horizon}")
    return np.stack(samples_x), np.stack(samples_y)
