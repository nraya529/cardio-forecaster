import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const CHANNEL_META = {
  hr: { label: "Heart rate", unit: "bpm", colour: "#f87171" },
  sbp: { label: "Systolic BP", unit: "mmHg", colour: "#60a5fa" },
  dbp: { label: "Diastolic BP", unit: "mmHg", colour: "#93c5fd" },
  spo2: { label: "SpO₂", unit: "%", colour: "#34d399" },
  resp: { label: "Respiration", unit: "rpm", colour: "#fbbf24" },
  temp: { label: "Temperature", unit: "°C", colour: "#c084fc" },
};

// minute 0 = "now". History is negative minutes, forecast is positive.
// Recharts wants observed/forecast as separate fields so the dashed line
// detects nulls and breaks correctly at the boundary.
function buildSeries(historyValues, forecastValues) {
  const total = historyValues.length + forecastValues.length;
  return Array.from({ length: total }, (_, i) => {
    const minute = i - historyValues.length;
    if (i < historyValues.length) {
      return { minute, observed: historyValues[i], forecast: null };
    }
    return { minute, observed: null, forecast: forecastValues[i - historyValues.length] };
  });
}

export function ForecastChart({ channel, historyValues, forecastValues }) {
  const meta = CHANNEL_META[channel];
  const data = buildSeries(historyValues, forecastValues);
  const stats = forecastValues.length
    ? {
        min: Math.min(...forecastValues).toFixed(1),
        max: Math.max(...forecastValues).toFixed(1),
        delta: (
          forecastValues[forecastValues.length - 1] - historyValues[historyValues.length - 1]
        ).toFixed(1),
      }
    : null;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
      <div className="mb-2 flex items-end justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-slate-500">{meta.label}</div>
          <div className="text-lg font-semibold" style={{ color: meta.colour }}>
            {historyValues[historyValues.length - 1].toFixed(1)} <span className="text-xs text-slate-400">{meta.unit}</span>
          </div>
        </div>
        {stats && (
          <div className="text-right text-xs text-slate-400">
            <div>forecast Δ {stats.delta} {meta.unit}</div>
            <div>range {stats.min} – {stats.max}</div>
          </div>
        )}
      </div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="minute" stroke="#475569" tick={{ fontSize: 10 }} />
            <YAxis stroke="#475569" tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                color: "#e2e8f0",
                fontSize: 12,
              }}
              labelFormatter={(v) => `t = ${v} min`}
            />
            <ReferenceLine x={0} stroke="#334155" strokeDasharray="2 4" label={{ value: "now", fill: "#94a3b8", fontSize: 10 }} />
            <Line
              type="monotone"
              dataKey="observed"
              stroke={meta.colour}
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="forecast"
              stroke={meta.colour}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
