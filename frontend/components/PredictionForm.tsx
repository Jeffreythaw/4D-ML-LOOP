"use client";

import { FormEvent, useState } from "react";

type PredictionFormProps = {
  loading: boolean;
  onSubmit: (drawNumber: number) => void;
};

export function PredictionForm({ loading, onSubmit }: PredictionFormProps) {
  const [drawNumber, setDrawNumber] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit(Number(drawNumber));
  }

  return (
    <form className="form" onSubmit={handleSubmit}>
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

      <p className="muted">
        The system predicts the next draw number from the selected base draw.
        Day type is auto-detected from the database.
      </p>

      <button className="button" type="submit" disabled={loading}>
        {loading ? "Loading..." : "Predict Next Draw"}
      </button>
    </form>
  );
}
