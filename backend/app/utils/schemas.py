"""Pydantic API contracts."""
from __future__ import annotations

from pydantic import BaseModel, Field


class VitalsWindow(BaseModel):
    hr: list[float] = Field(..., description="Heart rate (bpm)")
    sbp: list[float] = Field(..., description="Systolic blood pressure (mmHg)")
    dbp: list[float] = Field(..., description="Diastolic blood pressure (mmHg)")
    spo2: list[float] = Field(..., description="Peripheral oxygen saturation (%)")
    resp: list[float] = Field(..., description="Respiratory rate (breaths/min)")
    temp: list[float] = Field(..., description="Body temperature (°C)")

    def length(self) -> int:
        return len(self.hr)

    def is_uniform(self) -> bool:
        return len({len(getattr(self, ch)) for ch in ("hr", "sbp", "dbp", "spo2", "resp", "temp")}) == 1


class ForecastRequest(BaseModel):
    history: VitalsWindow
    patient_id: str | None = None


class ForecastChannel(BaseModel):
    channel: str
    values: list[float]


class ForecastResponse(BaseModel):
    patient_id: str | None
    horizon_minutes: int
    sampling_rate_hz: float
    channels: list[ForecastChannel]
    risk_series: list[float]
    risk_peak: float
    risk_peak_minute: int


class SimulationRequest(BaseModel):
    minutes: int = Field(60, ge=10, le=720)
    deterioration: bool | None = None
    seed: int | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
