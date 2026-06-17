# Deployment

## Phase One Frontend Deployment

Deploy only `frontend/` to Vercel.

```bash
cd frontend
npm install
npm run build
npx vercel
npx vercel --prod
```

Set this Vercel environment variable:

```text
NEXT_PUBLIC_API_BASE_URL=https://your-backend-host.example.com
```

## Backend Deployment

Do not deploy the FastAPI backend to Vercel in this phase.

Deploy `backend/` separately as a persistent Python service with:

- Python 3.11 or newer.
- Microsoft ODBC Driver for SQL Server installed.
- `pip install -r requirements.txt`.
- Environment variables from `backend/.env.example` supplied through the host secret manager.
- Runtime command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

The backend host must allow outbound network access to the existing remote SQL Server.

## Local Development

Backend:

```bash
cd backend
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```
