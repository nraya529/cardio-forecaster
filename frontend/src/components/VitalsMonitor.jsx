import { ForecastChart } from "./ForecastChart.jsx";

const CHANNELS = ["hr", "sbp", "dbp", "spo2", "resp", "temp"];

export function VitalsMonitor({ history, forecast }) {
  if (!history || !forecast) {
    return (
      <div className="grid h-64 place-items-center rounded-2xl border border-slate-800 bg-slate-900/40 text-slate-500">
        Awaiting inference…
      </div>
    );
  }
  const forecastByChannel = Object.fromEntries(
    forecast.channels.map((c) => [c.channel, c.values])
  );
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {CHANNELS.map((ch) => (
        <ForecastChart
          key={ch}
          channel={ch}
          historyValues={history.channels[ch]}
          forecastValues={forecastByChannel[ch] || []}
        />
      ))}
    </div>
  );
}
