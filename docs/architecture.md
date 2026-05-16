# Architecture

```
┌──────────────────────────┐
│   MIMIC-III Waveform DB  │  optional, credentialed (PhysioNet)
└────────────┬─────────────┘
             │ wfdb loader  ┐
             ▼              │  swap-in
┌──────────────────────────┐│  ┌────────────────────────┐
│ data/synthetic.py        │└─►│ ChannelStats z-scoring │
│  ICU-plausible signals   │    └────────────┬───────────┘
└────────────┬─────────────┘                 │
             │ (T, 6) float32                 ▼
             ▼                       ┌─────────────────────────────┐
┌──────────────────────────┐         │ Time-Series Transformer      │
│ streaming_window()       │ ───────►│ encoder×4 / decoder×2        │
│   history=360, horizon=120│         │ + risk head (BCE)            │
└──────────────────────────┘         └────────────┬─────────────────┘
                                                  │ autoregressive roll-out
                                                  ▼
                                       ┌────────────────────────┐
                                       │ FastAPI /forecast/*     │
                                       │ /simulate/*  /health    │
                                       └────────────┬────────────┘
                                                    │ JSON
                                                    ▼
                                       ┌────────────────────────┐
                                       │ React + Recharts UI     │
                                       │ live monitor, risk band │
                                       └────────────────────────┘
```

## Data layout

| Channel | Unit | Source column (MIMIC) |
| ------- | ---- | --------------------- |
| `hr`    | bpm  | HR / II               |
| `sbp`   | mmHg | ABPSys                |
| `dbp`   | mmHg | ABPDias               |
| `spo2`  | %    | SpO2 / %SpO2          |
| `resp`  | rpm  | Resp                  |
| `temp`  | °C   | Temp                  |

## Model

`TimeSeriesTransformer` (`backend/app/models/transformer.py`):

- Input projection: `Linear(C, d_model)`
- Sinusoidal positional encoding, persistent buffer
- Encoder: `nn.TransformerEncoder` with `n_encoder_layers` GELU layers
- Decoder: `nn.TransformerDecoder` with causal mask, teacher forcing during
  training and autoregressive roll-out at inference
- Two heads: `signal_head -> (B, H, C)` and `risk_head -> (B, H, 1)`

Training (`training/train.py`) optimises:

```
L = SmoothL1(signal_pred, signal_target) + λ · BCEWithLogits(risk, label)
```

with AdamW + warmup linear decay.

## Inference latency target

p95 `< 75 ms` on a single CPU forecast for `history=360, horizon=120, d_model=128`.
The autoregressive roll-out is the hot path; switch to non-autoregressive
forecasting (parallel decode) for sub-20 ms budgets.
