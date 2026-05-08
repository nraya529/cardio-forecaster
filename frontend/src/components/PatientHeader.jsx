export function PatientHeader({ patient, status }) {
  if (!patient) return null;
  const id = `MIMIC-SIM-${Math.abs(Math.round(patient.baseline_hr * 17))}`;
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-800 pb-4">
      <div>
        <div className="text-xs uppercase tracking-widest text-slate-500">Patient</div>
        <div className="text-xl font-semibold text-slate-100">{id}</div>
        <div className="text-sm text-slate-400">
          Age {Math.round(patient.age)} · Baseline HR {Math.round(patient.baseline_hr)} bpm ·
          BP {Math.round(patient.baseline_sbp)}/{Math.round(patient.baseline_dbp)} mmHg
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs">
        <span
          className={`h-2 w-2 rounded-full ${
            status === "ready"
              ? "bg-emerald-400"
              : status === "refreshing"
              ? "bg-cyan-400 animate-pulse"
              : status === "error"
              ? "bg-red-500"
              : "bg-amber-400 animate-pulse"
          }`}
        />
        <span className="uppercase tracking-wider text-slate-400">
          {status === "ready"
            ? "Live"
            : status === "refreshing"
            ? "Re-forecasting"
            : status === "error"
            ? "Offline"
            : "Connecting"}
        </span>
      </div>
    </div>
  );
}
