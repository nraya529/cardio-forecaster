# cardio-forecaster

Trying to teach a Transformer to forecast continuous-time ICU vital signs.

I got into this after reading Project Lumos's writeup on PhysioFM. The idea
that you'd model a whole patient as a continuous waveform — not a snapshot at
admission, not an APACHE score — made everything else I'd seen in clinical ML
feel weirdly low-resolution. This repo is me trying to build the smallest
useful piece of that.

## what it does

Eats six hours of multivariate vitals (HR, systolic + diastolic BP, SpO₂,
respiration, temperature, all at 1-minute resolution) and predicts the next
two hours, plus a per-minute "this trajectory looks bad" probability.

A FastAPI service does inference, a React dashboard plots the observed window
in solid color and the forecast as a dashed line, and there's a docker-compose
file that boots the whole thing locally.

![CI](https://github.com/nraya529/cardio-forecaster/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11-blue.svg)
![pytorch](https://img.shields.io/badge/pytorch-2.5-ee4c2c.svg)

## quickstart

```bash
docker compose up --build
```

- Backend Swagger: http://localhost:8000/docs
- Dashboard: http://localhost:5173

The backend boots even without a checkpoint — it just spins up an untrained
model so the API works end-to-end. If you want predictions that aren't random:

```bash
cd backend && pip install -r requirements-dev.txt
cd ..
python training/train.py --config training/config.yaml
```

That trains on synthetic ICU episodes. The real waveforms from MIMIC-III need
[PhysioNet credentialed access](https://physionet.org/content/mimic3wdb/1.0/)
which I'm still working through. The MIMIC loader is already wired up
(`backend/app/data/mimic_loader.py`) and is schema-compatible with the synthetic
generator, so swapping data sources won't touch the model or training code.

## the model

Encoder-decoder Transformer, basically the seq2seq skeleton from "Attention Is
All You Need" but with continuous targets. Two things are a bit different from
a vanilla setup:

1. **Continuous outputs.** Loss is SmoothL1 instead of cross-entropy. The
   decoder emits real-valued signals, not token logits.
2. **Two heads.** A signal head for the next-step prediction, and a risk head
   trained with BCE against per-minute deterioration labels from the data
   pipeline. They share the same decoder; the risk head is the cheap "while
   we're in here, also tell me if this is going sideways" bolt-on.

At inference I roll out autoregressively — each predicted step feeds back as
the next decoder input. Training uses teacher forcing so I don't pay the loop
cost in the backward pass.

```
history (B, 360, 6)
        │
        ▼
input projection + sinusoidal positional encoding
        │
        ▼
encoder × 4   ─►  memory  ──┐
                            │
last observed step ─► target proj + pos enc + causal mask
                            │
                            ▼
                        decoder × 2
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        signal_head (→ 6)        risk_head (→ 1)
```

I built a Bi-LSTM first (`backend/app/models/lstm.py`) as a sanity-floor
baseline. Transformer beats it on the 120-minute horizon, which roughly tracks
what people report in the time-series forecasting literature. Kept the LSTM in
the repo so it's easy to swap and re-measure.

## the dataset situation (honest version)

The synthetic generator (`backend/app/data/synthetic.py`) layers AR(1) noise
on top of per-channel population means, adds circadian terms with realistic
periods (~90 min for HR, ~120 min for BP, ~12 hr for temp), and occasionally
generates a "deterioration" window with the classic sepsis-like signature:
HR climbs, BP drops, SpO₂ falls, temp rises. Those windows get labeled so the
risk head has supervision.

It is **not** real data. I tried to make the synthetic distribution
physiologically defensible — parameters come from textbook references for ICU
populations, not vibes — but until I retrain on MIMIC waveforms the model is
solving a toy problem. The structure of the pipeline (loaders, normalization,
windowing, training loop) is real and would transfer; the trained weights are
not the point of this repo yet.

## API surface

`POST /api/v1/forecast/vitals` — give it six hours of every channel, get back
two hours of forecast plus a smoothed risk series and the projected
peak-risk minute.

```json
{
  "history": {
    "hr":   [/* 360 floats, one per minute */],
    "sbp":  [...],
    "dbp":  [...],
    "spo2": [...],
    "resp": [...],
    "temp": [...]
  },
  "patient_id": "MIMIC-01234"
}
```

`POST /api/v1/forecast/simulate` — lazy-mode endpoint that fabricates an
episode via the synthetic generator and forecasts its tail. The dashboard
polls this so the demo works without uploading anything.

Pydantic models are in `backend/app/utils/schemas.py`. The Swagger UI at
`/docs` is honestly the easiest way to explore them.

## frontend

React + Vite + Tailwind + Recharts. Dark UI because that's what real bedside
monitors look like, not because I think dark mode is cooler. Three preset
patients in the top bar (stable / deteriorating / random sample) so you can
flip between them and watch the risk indicator change. Charts plot the
observed window as a solid line and the forecast as a dashed continuation —
that visual distinction matters a lot when you're squinting at six
side-by-side panels.

## tests

```bash
cd backend && pytest -q
cd ../frontend && npm run build
```

Eleven pytest cases:

- synthetic generator shapes, dtypes, and that the deterioration window
  actually fires when forced
- streaming window correctly rejects episodes shorter than `history + horizon`
- channel normalization is a clean round-trip
- Transformer forward + autoregressive forecast emit the right shapes
- gradients actually propagate (caught a `register_buffer` bug this way)
- API contracts: simulate endpoint returns all six channels, forecast endpoint
  returns the full horizon, wrong-length payload returns 400

CI (`.github/workflows/ci.yml`) runs all of that plus the Vite build plus a
docker compose build on every push.

## layout

```
backend/
  app/
    api/         → routes.py
    core/        → settings + logging
    data/        → synthetic.py · mimic_loader.py · preprocessing.py
    models/      → transformer.py · lstm.py
    inference/   → predictor.py (model loading + post-processing)
    utils/       → schemas.py (pydantic)
  tests/         → test_data · test_model · test_api
  Dockerfile
frontend/
  src/
    components/  → VitalsMonitor · ForecastChart · RiskIndicator · PatientHeader
    hooks/       → useLiveVitals.js
    api/         → client.js
    App.jsx
  Dockerfile
training/
  train.py · evaluate.py · config.yaml
notebooks/       → exploration + a tiny training walkthrough
docs/            → architecture.md
docker-compose.yml
```

## things i still want to do

- [ ] swap the autoregressive decoder for parallel decoding — the loop is the
      main inference cost
- [ ] retrain on real MIMIC-III waveforms once my PhysioNet creds clear
- [ ] try Neural CDEs for the irregular-time variant (so I can train on the
      raw waveform sample rate instead of minute-resolution downsamples)
- [ ] calibration plots for the risk head — I don't trust the probabilities
      yet, the histogram of predicted risks is too peaky

## reading I leaned on

- Vaswani et al., *Attention Is All You Need* — encoder/decoder skeleton
- Lim & Zohren, *Time-series forecasting with deep learning: a survey* — good
  taxonomy for picking the architecture
- Johnson et al. on MIMIC-III — used for channel naming and population stats
- Lipton et al., *Modeling Missing Data in Clinical Time Series with RNNs* —
  what convinced me masked losses were not optional

---

built by [niketh raya](https://github.com/nraya529).
inspired by what [Project Lumos](https://www.projectlumos.org) is doing with
continuous-time physiological foundation models.
