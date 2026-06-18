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
      <section className="mode-card">
        <div>
          <h2>Current Prediction Mode</h2>
          <p className="muted">
            Automatically uses the latest completed base draw from SQL and predicts the next draw.
          </p>
        </div>

        <button className="button" type="button" disabled={loading} onClick={onCurrentPrediction}>
          {loading ? "Loading..." : "Predict Current Next Draw"}
        </button>
      </section>

      <section className="mode-card">
        <div>
          <h2>Historical Audit Mode</h2>
          <p className="muted">
            Enter any base draw number to reproduce a locked Top 5 for the next draw, then verify
            through the SQL firewall.
          </p>
        </div>

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
            {loading ? "Loading..." : "Run Historical Audit"}
          </button>
        </form>
      </section>
    </div>
  );
}
