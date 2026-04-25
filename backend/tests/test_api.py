from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


def _client() -> TestClient:
    get_settings.cache_clear()
    return TestClient(create_app())


def test_health_endpoint_reports_ok():
    with _client() as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "device" in body


def test_simulate_episode_returns_channels():
    with _client() as client:
        response = client.post("/api/v1/simulate/episode", json={"minutes": 30, "seed": 7})
        assert response.status_code == 200
        body = response.json()
        assert set(body["channels"]) == {"hr", "sbp", "dbp", "spo2", "resp", "temp"}
        for series in body["channels"].values():
            assert len(series) == 30 * 60


def test_forecast_simulated_returns_full_horizon():
    settings = get_settings()
    with _client() as client:
        response = client.post(
            "/api/v1/forecast/simulate",
            json={"minutes": settings.history_window + 60, "seed": 42},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["horizon_minutes"] == settings.forecast_horizon
        assert len(body["risk_series"]) == settings.forecast_horizon
        assert len(body["channels"]) == len(settings.channel_names)
        for channel in body["channels"]:
            assert len(channel["values"]) == settings.forecast_horizon


def test_forecast_rejects_wrong_length():
    settings = get_settings()
    payload = {
        "history": {ch: [0.0] * (settings.history_window - 1) for ch in settings.channel_names},
        "patient_id": "test",
    }
    with _client() as client:
        response = client.post("/api/v1/forecast/vitals", json=payload)
        assert response.status_code == 400
