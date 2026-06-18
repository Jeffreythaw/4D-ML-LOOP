#!/bin/zsh
set -e

PROJECT_ROOT="/Users/kojeffrey/4D-ML-LOOP"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Jeffrey Quad Engine v2 - Local App Launcher"
echo "============================================================"
echo "Project: $PROJECT_ROOT"
echo "Backend: http://127.0.0.1:8000"
echo "Frontend: http://localhost:3000"
echo "Logs: $LOG_DIR"
echo "============================================================"

cd "$PROJECT_ROOT"

if [ ! -f "$BACKEND_DIR/.env" ]; then
  echo "ERROR: backend/.env not found."
  echo "Create backend/.env before launching the app."
  exit 1
fi

if [ ! -f "$FRONTEND_DIR/.env.local" ]; then
  echo "Creating frontend/.env.local"
  cat > "$FRONTEND_DIR/.env.local" <<ENVEOF
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
ENVEOF
fi

if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Backend port 8000 is already in use. Leaving existing process running."
else
  echo "Starting FastAPI backend..."
  cd "$BACKEND_DIR"
  nohup "$PROJECT_ROOT/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > "$LOG_DIR/backend.log" 2>&1 &
  echo $! > "$LOG_DIR/backend.pid"
fi

cd "$PROJECT_ROOT"

if lsof -iTCP:3000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Frontend port 3000 is already in use. Leaving existing process running."
else
  echo "Starting Next.js frontend..."
  cd "$FRONTEND_DIR"
  nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$LOG_DIR/frontend.pid"
fi

echo "Waiting for services..."
sleep 4

open "http://localhost:3000"

echo "============================================================"
echo "Launcher completed."
echo "Open: http://localhost:3000"
echo "Backend log: $LOG_DIR/backend.log"
echo "Frontend log: $LOG_DIR/frontend.log"
echo "============================================================"
