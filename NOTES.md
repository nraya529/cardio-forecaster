# notes

scratch file. things i learned, things i tried, things to come back to.

## 2026-04 — picking the architecture

started with a stacked LSTM because it was easier to reason about. worked
fine for ~30 min horizon, degraded fast after that. classic "the LSTM forgets
what the BP was doing 4 hours ago even though that's actually relevant."

switched to a Transformer. the encoder/decoder split is what i wanted because
it gives the decoder access to the *whole* history through cross-attention,
not just a final hidden state. the BP trend from 5 hours ago matters when
predicting deterioration.

kept the LSTM around as a baseline. transformer beats it on horizon=120 by a
margin big enough that i believe the result.

## the autoregressive vs parallel decoder thing

right now the decoder rolls out one step at a time at inference. each step
needs all the previous steps as input. that's `O(horizon)` forward passes and
it's slow.

the alternative is parallel decoding: emit all 120 steps in one shot from
some learned query tokens. faster, but i wasn't sure it'd work as well for
multivariate signals with strong autocorrelation. punted on it. would be the
first thing i'd try to make inference fast enough for an actual hospital
deployment.

## positional encoding bug

burned a few hours on this. i was registering the positional encoding as a
plain attribute, not a buffer. it worked fine on cpu but the moment i moved
the model to gpu the PE was still on cpu and i got a device mismatch error
buried five frames deep in a transformer layer.

fix: `self.register_buffer("pe", pe.unsqueeze(0), persistent=False)`

`persistent=False` means it won't get saved in the state dict, which is fine
because it's deterministic and we can recreate it on load. also keeps the
checkpoint smaller.

## risk head supervision

the synthetic generator emits per-minute binary labels for the deterioration
window. easy supervision. for real MIMIC data i'd need to define what counts
as deterioration — sepsis-3, MEWS score crossing a threshold, transfer to
higher acuity care, something. punting on that until i have the real data.

## things that didn't work

- tried huber loss instead of smooth-l1. basically identical, no reason to
  prefer it.
- tried adding a learned positional encoding on top of the sinusoidal one.
  no improvement, just more parameters. dropped it.
- tried teacher-forcing-free training (always autoregress). way slower per
  epoch, also slightly worse on val. teacher forcing it is.

## things i would do differently

if i were starting over i'd build the data pipeline before the model.
spent half a day chasing what i thought was a model bug that turned out to
be a `streaming_window` off-by-one error. data pipeline tests catch this
in 100ms.

also: the `ChannelStats` thing — i originally stored mean/std as a json
sidecar file. moved them into the checkpoint so loading is one step.
should have done that from the start.

## todos i actually plan to do

- [ ] write a real benchmark harness that measures p50/p95 inference
      latency across batch sizes
- [ ] sweep dropout and warmup_steps once i have MIMIC. on synthetic the
      val loss is too flat to learn anything from hyperparam tuning.
- [ ] make the frontend show the confidence interval, not just the mean
      forecast. recharts supports a banded area chart.
