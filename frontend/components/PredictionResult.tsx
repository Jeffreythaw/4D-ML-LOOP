"use client";

import type { PredictionResponse } from "../lib/api";

type PredictionResultProps = {
  result: PredictionResponse | null;
  mode: "current" | "historical" | null;
  verifying: boolean;
  onVerify: () => void;
};

export function PredictionResult({ result, mode, verifying, onVerify }: PredictionResultProps) {
  if (!result) {
    return (
      <section className="result result-empty" aria-live="polite">
        <div className="empty-orbit" aria-hidden="true">
          <span />
        </div>
        <div>
          <h2>Prediction workspace ready</h2>
          <p className="muted">
            Run Current Prediction or a Historical Audit to populate the secured Top 5 result.
          </p>
        </div>
      </section>
    );
  }

  const targetDraw = result.target_draw_number ?? result.draw_number + 1;
  const dayType = result.day_type ?? "auto-detected";
  const engineSource =
    result.predictions.find((prediction) => prediction.source)?.source ?? "existing-engine-wrapper";
  const isTemporalMaster = engineSource === "E1_TEMPORAL_CONTEXT_MATCH";

  return (
    <section className="result" aria-live="polite">
      <div className="result-header">
        <div>
          <span className="result-kicker">Secured output</span>
          <h2>Top 5 Predictions</h2>
        </div>

        {mode === "historical" ? (
          <button
            className="button secondary"
            type="button"
            onClick={onVerify}
            disabled={verifying || result.predictions.length === 0}
          >
            {verifying ? "Verifying..." : "Verify"}
          </button>
        ) : null}
      </div>

      <dl className="result-meta">
        <div>
          <dt>Base Draw</dt>
          <dd>{result.draw_number}</dd>
        </div>
        <div>
          <dt>Target Draw</dt>
          <dd>{targetDraw}</dd>
        </div>
        <div>
          <dt>Day Type</dt>
          <dd>{dayType}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd className="verification-status">{result.verification_status}</dd>
        </div>
        <div className="engine-meta">
          <dt>Engine</dt>
          <dd className={isTemporalMaster ? "engine-badge master" : "engine-badge"}>
            <span>{engineSource}</span>
            {isTemporalMaster ? <small>Master Engine</small> : null}
          </dd>
        </div>
      </dl>

      <ul className="prediction-list">
        {result.predictions.map((prediction) => (
          <li key={`${prediction.rank}-${prediction.number}`}>
            <span className="rank">{prediction.rank}</span>
            <span className="number">{prediction.number}</span>
            <span className="source">
              {prediction.source === "E1_TEMPORAL_CONTEXT_MATCH"
                ? "Temporal Context"
                : prediction.source ?? "Existing Engine"}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
