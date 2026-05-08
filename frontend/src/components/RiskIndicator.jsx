import clsx from "clsx";

export function RiskIndicator({ risk, peakMinute }) {
  const level = risk > 0.66 ? "high" : risk > 0.33 ? "mid" : "low";
  const label = { low: "STABLE", mid: "WATCH", high: "CRITICAL" }[level];
  const colour = {
    low: "text-risk-low",
    mid: "text-risk-mid",
    high: "text-risk-high",
  }[level];

  return (
    <div className="flex items-center gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
      <span className={clsx("pulse-dot h-3 w-3 rounded-full", colour, `bg-current`)} aria-hidden />
      <div>
        <div className="text-xs uppercase tracking-widest text-slate-500">Deterioration probability</div>
        <div className="flex items-baseline gap-3">
          <span className={clsx("text-3xl font-semibold", colour)}>{(risk * 100).toFixed(1)}%</span>
          <span className={clsx("text-sm font-semibold", colour)}>{label}</span>
        </div>
        <div className="text-xs text-slate-400">
          Peak risk forecast in <span className="text-slate-200">~{peakMinute} min</span>
        </div>
      </div>
    </div>
  );
}
