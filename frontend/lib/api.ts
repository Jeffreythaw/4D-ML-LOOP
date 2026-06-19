const configuredApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");
const API_BASE_URL =
  configuredApiBaseUrl ??
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

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

export type LatestDrawResponse = {
  draw_number: number;
  target_draw_number: number;
  draw_date: string;
  day_type?: string | null;
};

export type VerificationResponse = {
  draw_number: number;
  day_type?: string | null;
  verification_status: string;
  hit_count: number;
  details: Record<string, unknown>;
};


export async function getLatestDraw(): Promise<LatestDrawResponse> {
  assertApiBaseUrlConfigured();
  const response = await fetch(`${API_BASE_URL}/api/latest-draw`);

  if (!response.ok) {
    const detail = await readError(response);
    throw new Error(detail);
  }

  return response.json() as Promise<LatestDrawResponse>;
}

export async function predict(payload: {
  draw_number: number;
  mode?: "Current" | "Historical";
}): Promise<PredictionResponse> {
  return postJson<PredictionResponse>("/api/predict", payload);
}

export async function verify(payload: {
  draw_number: number;
  source_draw_number?: number | null;
  mode?: "Current" | "Historical";
  day_type?: string | null;
  predictions: PredictionCandidate[];
}): Promise<VerificationResponse> {
  return postJson<VerificationResponse>("/api/verify", payload);
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  assertApiBaseUrlConfigured();
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

function assertApiBaseUrlConfigured(): void {
  if (!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL is not configured.");
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : "Request failed.";
  } catch {
    return "Request failed.";
  }
}
