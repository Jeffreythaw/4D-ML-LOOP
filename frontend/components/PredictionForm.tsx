"use client";

import { FormEvent, useState } from "react";

type PredictionFormProps = {
  loading: boolean;
  onCurrentPrediction: () => void;
  onHistoricalAudit: (drawNumber: number) => void;
};

export function PredictionForm({
  loading,
  onCurrentPrediction,
  onHistoricalAudit,
}: PredictionFormProps) {
  const [drawNumber, setDrawNumber] = useState("");

  function handleHistoricalSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onHistoricalAudit(Number(drawNumber));
  }

  return (
    <div className="form">
      <section className="mode-card mode-card-primary">
        <div className="mode-card-copy">
          <div className="mode-icon mode-icon-primary" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 3v4m0 10v4M3 12h4m10 0h4M5.6 5.6l2.8 2.8m7.2 7.2 2.8 2.8m0-12.8-2.8 2.8m-7.2 7.2-2.8 2.8" />
              <circle cx="12" cy="12" r="3.2" />
            </svg>
          </div>
          <div>
            <span className="mode-kicker">Live production</span>
            <h2>Current Prediction Mode</h2>
          </div>
        </div>

        <p className="muted mode-description">
          Automatically uses the latest completed base draw from SQL and predicts the next draw.
        </p>

        <button
          className="button primary-action"
          type="button"
          disabled={loading}
          onClick={onCurrentPrediction}
        >
          <span>{loading ? "Loading prediction..." : "Predict Current Next Draw"}</span>
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="m9 18 6-6-6-6" />
          </svg>
        </button>
      </section>

      <section className="mode-card mode-card-secondary">
        <div className="mode-card-copy">
          <div className="mode-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
              <path d="M3 3v5h5M12 7v5l3 2" />
            </svg>
          </div>
          <h2>Historical Audit Mode</h2>
        </div>

        <p className="muted mode-description">
          Enter any base draw number to reproduce a locked Top 5, then verify through the SQL
          firewall.
        </p>

        <form className="inline-form" onSubmit={handleHistoricalSubmit}>
          <label className="field">
            <span>Base Draw No</span>
            <input
              min="1"
              required
              type="number"
              value={drawNumber}
              onChange={(event) => setDrawNumber(event.target.value)}
              placeholder="5486"
            />
          </label>

          <button className="button secondary" type="submit" disabled={loading}>
            <span>{loading ? "Loading audit..." : "Run Historical Audit"}</span>
            <svg aria-hidden="true" viewBox="0 0 24 24">
              <path d="m9 18 6-6-6-6" />
            </svg>
          </button>
        </form>
      </section>
    </div>
  );
}
