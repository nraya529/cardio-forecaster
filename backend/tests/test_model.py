import torch

from app.models.transformer import TimeSeriesTransformer, TransformerConfig


def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        n_channels=6,
        history=32,
        horizon=8,
        d_model=32,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )


def test_teacher_forced_forward_shapes():
    config = _tiny_config()
    model = TimeSeriesTransformer(config).eval()
    history = torch.zeros(2, config.history, config.n_channels)
    target = torch.zeros(2, config.horizon, config.n_channels)
    signal, risk = model(history, target)
    assert signal.shape == (2, config.horizon, config.n_channels)
    assert risk.shape == (2, config.horizon, 1)


def test_autoregressive_forecast_shapes():
    config = _tiny_config()
    model = TimeSeriesTransformer(config).eval()
    history = torch.randn(3, config.history, config.n_channels)
    signal, risk = model.autoregressive_forecast(history)
    assert signal.shape == (3, config.horizon, config.n_channels)
    assert risk.shape == (3, config.horizon)
    assert torch.all(risk >= 0) and torch.all(risk <= 1)


def test_loss_backprop_runs():
    config = _tiny_config()
    model = TimeSeriesTransformer(config)
    history = torch.randn(2, config.history, config.n_channels)
    target = torch.randn(2, config.horizon, config.n_channels)
    pred, _ = model(history, target)
    loss = torch.nn.functional.smooth_l1_loss(pred, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)
