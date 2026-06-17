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
