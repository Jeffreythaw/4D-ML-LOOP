const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type PredictionCandidate = {
  rank: number;
  number: string;
  score?: number | null;
  source?: string | null;
};

export type PredictionResponse = {
  draw_number: number;
  target_draw_number?: number | null;
  day_type?: string | null;
  predictions: PredictionCandidate[];
  verification_status: string;
};

export type VerificationResponse = {
  draw_number: number;
  day_type?: string | null;
  verification_status: string;
  hit_count: number;
  details: Record<string, unknown>;
};

export async function predict(payload: {
  draw_number: number;
}): Promise<PredictionResponse> {
  return postJson<PredictionResponse>("/api/predict", payload);
}

export async function verify(payload: {
  draw_number: number;
  day_type?: string | null;
  predictions: PredictionCandidate[];
}): Promise<VerificationResponse> {
  return postJson<VerificationResponse>("/api/verify", payload);
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await readError(response);
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : "Request failed.";
  } catch {
    return "Request failed.";
  }
}
