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

  return (
    <section className="result" aria-live="polite">
      <div className="result-header">
        <div>
          <h2>Top 5 predictions</h2>
          <p className="muted">
            Draw {result.draw_number} - {result.day_type} - {result.verification_status}
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
