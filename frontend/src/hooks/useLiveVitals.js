import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";

const CHANNELS = ["hr", "sbp", "dbp", "spo2", "resp", "temp"];

function decimate(series, target = 360) {
  if (series.length <= target) return series;
  const step = series.length / target;
  return Array.from({ length: target }, (_, i) => series[Math.floor(i * step)]);
}

export function useLiveVitals({ seed = 42, deterioration = null, refreshMs = 12000 } = {}) {
  const [history, setHistory] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState(null);
  const cursorRef = useRef(0);

  const fetchOnce = useCallback(async () => {
    try {
      setStatus((s) => (s === "ready" ? "refreshing" : "loading"));
      const ep = await api.simulateEpisode({ minutes: 720, seed, deterioration });
      const fc = await api.forecastSimulate({ minutes: 480, seed, deterioration });
      const compact = {};
      CHANNELS.forEach((ch) => {
        compact[ch] = decimate(ep.channels[ch], 360);
      });
      setHistory({ channels: compact, patient: ep.patient });
      setForecast(fc);
      setStatus("ready");
      cursorRef.current = (cursorRef.current + 1) % 1000;
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }, [seed, deterioration]);

  useEffect(() => {
    fetchOnce();
    const id = setInterval(fetchOnce, refreshMs);
    return () => clearInterval(id);
  }, [fetchOnce, refreshMs]);

  return { history, forecast, status, error, refresh: fetchOnce };
}
