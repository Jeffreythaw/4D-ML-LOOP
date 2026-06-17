# Jeffrey Quad Engine v2 Architecture

## Target Shape

- Frontend: React / Next.js, deployed to Vercel in phase one.
- Backend: Python FastAPI, deployed separately as a persistent Python service.
- Database: Existing remote SQL Server.
- ML Engine: Existing Python NumPy Step 2 / Step 3 scripts.

## Boundaries

The frontend calls the FastAPI backend through `NEXT_PUBLIC_API_BASE_URL`.

The backend exposes API routes and delegates prediction generation to `backend/app/core/ml_adapter.py`. The adapter is a wrapper boundary around the existing Step 2 and Step 3 scripts, not a replacement for the engine.

Prediction verification must go through the existing SQL Server stored procedure configured by `SQL_VERIFY_PROCEDURE`. The backend must not directly query hidden target winners.

## Current Integration Status

The skeleton is wired for HTTP, schemas, CORS, SQL environment configuration, and verification procedure calls. The prediction adapter intentionally raises a clear service error until a safe single-draw public entry point is exposed by the existing Step 3 script.
