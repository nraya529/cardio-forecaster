import { useState } from "react";
import { PatientHeader } from "./components/PatientHeader.jsx";
import { RiskIndicator } from "./components/RiskIndicator.jsx";
import { VitalsMonitor } from "./components/VitalsMonitor.jsx";
import { useLiveVitals } from "./hooks/useLiveVitals.js";

const PRESETS = [
  { id: "stable", label: "Stable patient", seed: 11, deterioration: false },
  { id: "deterioration", label: "Onset deterioration", seed: 73, deterioration: true },
  { id: "random", label: "Population sample", seed: 1337, deterioration: null },
];

export default function App() {
  const [preset, setPreset] = useState(PRESETS[1]);
  const { history, forecast, status, error, refresh } = useLiveVitals({
    seed: preset.seed,
    deterioration: preset.deterioration,
  });

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <header className="mb-6 flex flex-col gap-1">
        <span className="text-xs uppercase tracking-[0.4em] text-cyan-400">
          continuous physiological forecasting
        </span>
        <h1 className="text-3xl font-semibold text-slate-100">CardioForecaster</h1>
        <p className="max-w-2xl text-sm text-slate-400">
          Multivariate Time-Series Transformer trained on continuous-time vital-sign waveforms.
          Forecasts the next two hours of heart rate, blood pressure, SpO₂, respiration and
          temperature, and emits a per-minute clinical deterioration probability.
        </p>
      </header>

      <div className="mb-6 flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            onClick={() => setPreset(p)}
            className={`rounded-full border px-4 py-1.5 text-xs uppercase tracking-wider transition ${
              preset.id === p.id
                ? "border-cyan-400 bg-cyan-400/10 text-cyan-200"
                : "border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200"
            }`}
          >
            {p.label}
          </button>
        ))}
        <button
          onClick={refresh}
          className="ml-auto rounded-full border border-slate-700 px-4 py-1.5 text-xs uppercase tracking-wider text-slate-300 hover:border-slate-500"
        >
          Re-forecast
        </button>
      </div>

      <PatientHeader patient={history?.patient} status={status} />

      {error && (
        <div className="my-4 rounded-xl border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-200">
          {error}
        </div>
      )}

      <div className="my-6 grid gap-4 md:grid-cols-3">
        <RiskIndicator risk={forecast?.risk_peak || 0} peakMinute={forecast?.risk_peak_minute || 0} />
        <Stat
          label="History window"
          value={`${history ? history.channels.hr.length : 0} min`}
          sub="downsampled for display"
        />
        <Stat
          label="Forecast horizon"
          value={`${forecast?.horizon_minutes || 0} min`}
          sub={`@ ${forecast?.sampling_rate_hz?.toFixed?.(1) || "1.0"} Hz`}
        />
      </div>

      <VitalsMonitor history={history} forecast={forecast} />

      <footer className="mt-8 text-xs text-slate-500">
        Solid line = observed window · Dashed line = autoregressive Transformer forecast.
        Model trained on synthetic ICU episodes; swap in the MIMIC-III Waveform Database via{" "}
        <code className="rounded bg-slate-900 px-1 py-0.5">backend/app/data/mimic_loader.py</code>{" "}
        once credentialed.
      </footer>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <div className="text-xs uppercase tracking-widest text-slate-500">{label}</div>
      <div className="text-2xl font-semibold text-slate-100">{value}</div>
      {sub && <div className="text-xs text-slate-400">{sub}</div>}
    </div>
  );
}
