"use client";

import { FormEvent, useState } from "react";

type PredictionFormProps = {
  loading: boolean;
  onSubmit: (drawNumber: number, dayType: string) => void;
};

const dayTypes = ["Wednesday", "Saturday", "Sunday", "Special"];

export function PredictionForm({ loading, onSubmit }: PredictionFormProps) {
  const [drawNumber, setDrawNumber] = useState("");
  const [dayType, setDayType] = useState(dayTypes[0]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit(Number(drawNumber), dayType);
  }

  return (
    <form className="form" onSubmit={handleSubmit}>
      <label className="field">
        <span>Draw number</span>
        <input
          min="1"
          required
          type="number"
          value={drawNumber}
          onChange={(event) => setDrawNumber(event.target.value)}
          placeholder="4051"
        />
      </label>

      <label className="field">
        <span>Day type</span>
        <select value={dayType} onChange={(event) => setDayType(event.target.value)}>
          {dayTypes.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </label>

      <button className="button" type="submit" disabled={loading}>
        {loading ? "Loading..." : "Predict"}
      </button>
    </form>
  );
}
