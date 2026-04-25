"""HTTP routes."""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.data.synthetic import generate_episode
from app.inference.predictor import CardioForecaster
from app.utils.schemas import (
    ForecastChannel,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    SimulationRequest,
    VitalsWindow,
)

logger = logging.getLogger(__name__)

router = APIRouter()
_predictor: CardioForecaster | None = None


def get_predictor() -> CardioForecaster:
    global _predictor
    if _predictor is None:
        _predictor = CardioForecaster()
        _predictor.load()
    return _predictor


@router.get("/health", response_model=HealthResponse)
def health(predictor: CardioForecaster = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=predictor.model is not None,
        device=str(predictor.device),
    )


@router.post("/forecast/vitals", response_model=ForecastResponse)
def forecast_vitals(
    request: ForecastRequest,
    predictor: CardioForecaster = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> ForecastResponse:
    if not request.history.is_uniform():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All vital-sign channels must have the same length.",
        )
    if request.history.length() != settings.history_window:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"History must contain {settings.history_window} samples per channel, "
                f"got {request.history.length()}."
            ),
        )
    history = np.stack(
        [getattr(request.history, ch) for ch in settings.channel_names],
        axis=1,
    ).astype(np.float32)
    result = predictor.forecast(history)
    return ForecastResponse(
        patient_id=request.patient_id,
        horizon_minutes=settings.forecast_horizon,
        sampling_rate_hz=settings.sampling_rate_hz,
        channels=[
            ForecastChannel(channel=name, values=result.signals[:, idx].tolist())
            for idx, name in enumerate(settings.channel_names)
        ],
        risk_series=result.risk.tolist(),
        risk_peak=result.risk_peak,
        risk_peak_minute=result.risk_peak_minute,
    )


@router.post("/simulate/episode")
def simulate_episode(
    request: SimulationRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    signals, labels, patient = generate_episode(
        n_minutes=request.minutes,
        sampling_rate_hz=settings.sampling_rate_hz,
        deterioration=request.deterioration,
        seed=request.seed,
    )
    return {
        "channels": {
            name: signals[:, idx].tolist()
            for idx, name in enumerate(settings.channel_names)
        },
        "deterioration_labels": labels.tolist(),
        "patient": {
            "age": patient.age,
            "baseline_hr": patient.baseline_hr,
            "baseline_sbp": patient.baseline_sbp,
            "baseline_dbp": patient.baseline_dbp,
            "baseline_spo2": patient.baseline_spo2,
            "baseline_resp": patient.baseline_resp,
            "baseline_temp": patient.baseline_temp,
            "deterioration_prob": patient.deterioration_prob,
        },
    }


@router.post("/forecast/simulate", response_model=ForecastResponse)
def forecast_simulated(
    request: SimulationRequest,
    predictor: CardioForecaster = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> ForecastResponse:
    """Convenience endpoint: simulate an episode and forecast its tail.

    The dashboard polls this so demos work end-to-end without uploading data.
    """
    needed = settings.history_window + 1
    minutes = max(request.minutes, needed)
    signals, _, _ = generate_episode(
        n_minutes=minutes,
        sampling_rate_hz=settings.sampling_rate_hz,
        deterioration=request.deterioration,
        seed=request.seed,
    )
    history = signals[-settings.history_window:]
    history_window = VitalsWindow(
        **{name: history[:, idx].tolist() for idx, name in enumerate(settings.channel_names)}
    )
    return forecast_vitals(
        ForecastRequest(history=history_window, patient_id=f"sim-{request.seed or 'random'}"),
        predictor=predictor,
        settings=settings,
    )
