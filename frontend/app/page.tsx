"use client";

import { useState } from "react";

import { PredictionForm } from "../components/PredictionForm";
import { PredictionResult } from "../components/PredictionResult";
import { getLatestDraw, predict, verify } from "../lib/api";
import type { PredictionResponse } from "../lib/api";

export default function Home() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [mode, setMode] = useState<"current" | "historical" | null>(null);
  const [loading, setLoading] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCurrentPrediction() {
    setLoading(true);
    setError(null);
    setResult(null);
    setMode("current");

    try {
      const latestDraw = await getLatestDraw();
      setResult(await predict({ draw_number: latestDraw.draw_number, mode: "Current" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Current prediction request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleHistoricalAudit(drawNumber: number) {
    setLoading(true);
    setError(null);
    setResult(null);
    setMode("historical");

    try {
      setResult(await predict({ draw_number: drawNumber, mode: "Historical" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Historical audit request failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleVerify() {
    if (!result) return;

    setVerifying(true);
    setError(null);

    try {
      const verification = await verify({
        draw_number: result.target_draw_number ?? result.draw_number + 1,
        source_draw_number: result.draw_number,
        mode: "Historical",
        day_type: result.day_type,
        predictions: result.predictions,
      });

      setResult({
        ...result,
        verification_status: `${verification.verification_status} (${verification.hit_count} hits)`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification request failed.");
    } finally {
      setVerifying(false);
    }
  }

  return (
    <main className="shell">
      <section className="panel">
        <div className="heading">
          <h1>Jeffrey Quad Engine v2</h1>
          <p>Local research dashboard for read-only prediction and SQL firewall verification.</p>
        </div>

        <PredictionForm
          onCurrentPrediction={handleCurrentPrediction}
          onHistoricalAudit={handleHistoricalAudit}
          loading={loading}
        />

        {mode ? (
          <p className="muted mode-label">
            Mode: {mode === "current" ? "Current Prediction" : "Historical Audit"}
          </p>
        ) : null}

        {error ? <p className="error">{error}</p> : null}

        <PredictionResult result={result} mode={mode} onVerify={handleVerify} verifying={verifying} />
      </section>
    </main>
  );
}
