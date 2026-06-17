# Jeffrey Quad Engine v2 Frontend

Next.js frontend for the Jeffrey Quad Engine v2 FastAPI backend.

## Local Development

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` to the deployed backend URL when using a remote API.

## Vercel Deployment

Deploy only this frontend folder in phase one. Configure the Vercel project root as `frontend` and set `NEXT_PUBLIC_API_BASE_URL` to the separately deployed FastAPI backend URL.
