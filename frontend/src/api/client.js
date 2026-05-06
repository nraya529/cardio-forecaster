const BASE = "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  simulateEpisode: ({ minutes = 60, seed = null, deterioration = null } = {}) =>
    request("/simulate/episode", {
      method: "POST",
      body: JSON.stringify({ minutes, seed, deterioration }),
    }),
  forecastSimulate: ({ minutes = 480, seed = null, deterioration = null } = {}) =>
    request("/forecast/simulate", {
      method: "POST",
      body: JSON.stringify({ minutes, seed, deterioration }),
    }),
  forecastVitals: (history, patientId = null) =>
    request("/forecast/vitals", {
      method: "POST",
      body: JSON.stringify({ history, patient_id: patientId }),
    }),
};
