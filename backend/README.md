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
with the repository root as Docker context because the API imports the approved
Step 2 and Step 3 engine modules from the repository root.

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
