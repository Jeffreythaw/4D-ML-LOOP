# Jeffrey Quad Engine v2 Backend

FastAPI integration boundary for the existing Jeffrey Quad Engine v2 Python scripts.

This backend does not rewrite the ML engine and does not read hidden target winners. Prediction generation is isolated behind `app/core/ml_adapter.py`; verification is isolated behind the SQL Server stored procedure configured by `SQL_VERIFY_PROCEDURE`.

## Local Development

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Set real SQL Server credentials only in `.env` or the deployment platform secret manager. Do not commit `.env`.

## Routes

- `GET /health`
- `POST /api/predict`
- `POST /api/verify`

## Backend Deployment

Do not deploy this Python service to Vercel in phase one. Deploy it separately as a persistent Python service with the required ODBC driver installed and environment variables supplied by the host.

For Render, use the repository-root `render.yaml`. It builds `backend/Dockerfile`
with `backend/` as Docker context. Tracked deployment copies of the approved
Step 2 and Step 3 engine modules live in `backend/` so the service remains
self-contained when Render configures the service root directory as `backend`.

When either root engine module changes, update its matching backend
`*.py.source` deployment copy before release and verify both pairs with `cmp`.

The backend image also includes the validated 40-year knowledge pack:

- `/app/full_history_live_knowledge.py`
- `/app/live_knowledge/full_history_engine_pack.json`

Set `J4D_FULL_HISTORY_KNOWLEDGE_ENABLED=0` to fail back to the existing engines.
The pack is only eligible when the source DrawNo is at or beyond its training
cutoff, preventing future-data leakage during historical replay.
The Docker image restores the `.py` filenames under `/app` so the existing
runtime imports remain unchanged.

Required Render environment variables:

- `SQL_SERVER_HOST`
- `SQL_SERVER_PORT` (normally `1433`; omit it if the host already embeds a port)
- `SQL_SERVER_DATABASE`
- `SQL_SERVER_USERNAME`
- `SQL_SERVER_PASSWORD`
- `SQL_SERVER_DRIVER` (`ODBC Driver 18 for SQL Server`)
- `SQL_ENCRYPT`
- `SQL_TRUST_SERVER_CERTIFICATE`
- `SQL_VERIFY_PROCEDURE`
- `FRONTEND_URL` (comma-separated exact HTTPS origins allowed by CORS)

Enter credentials only in the Render Dashboard. Never commit a production `.env`.
