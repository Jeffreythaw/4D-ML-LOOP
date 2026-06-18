"use client";

import type { PredictionResponse } from "../lib/api";

type PredictionResultProps = {
  result: PredictionResponse | null;
  verifying: boolean;
  onVerify: () => void;
};

export function PredictionResult({ result, verifying, onVerify }: PredictionResultProps) {
  if (!result) {
    return <p className="muted result">No prediction requested yet.</p>;
  }

  const targetDraw = result.target_draw_number ?? result.draw_number + 1;
  const dayType = result.day_type ?? "auto-detected";

  return (
    <section className="result" aria-live="polite">
      <div className="result-header">
        <div>
          <h2>Top 5 predictions</h2>
          <p className="muted">
            Base Draw {result.draw_number} → Predicting Draw {targetDraw}
          </p>
          <p className="muted">
            Day Type: {dayType} · Status: {result.verification_status}
          </p>
        </div>

        <button
          className="button secondary"
          type="button"
          onClick={onVerify}
          disabled={verifying || result.predictions.length === 0}
        >
          {verifying ? "Verifying..." : "Verify"}
        </button>
      </div>

      <ul className="prediction-list">
        {result.predictions.map((prediction) => (
          <li key={`${prediction.rank}-${prediction.number}`}>
            <span className="rank">#{prediction.rank}</span>
            <span className="number">{prediction.number}</span>
            <span className="source">{prediction.source ?? "existing-engine-wrapper"}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
