#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Users/kojeffrey/4D-ML-LOOP"

echo "============================================================"
echo " Jeffrey Quad Engine v2 - Local App Stopper"
echo "============================================================"

stop_port() {
  local port="$1"
  local label="$2"

  local pids
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN || true)

  if [ -z "$pids" ]; then
    echo "$label port $port: no running process found."
    return 0
  fi

  echo "$label port $port: stopping PID(s): $pids"
  echo "$pids" | xargs kill

  sleep 1

  local remaining
  remaining=$(lsof -tiTCP:"$port" -sTCP:LISTEN || true)

  if [ -n "$remaining" ]; then
    echo "$label port $port: force stopping remaining PID(s): $remaining"
    echo "$remaining" | xargs kill -9
  fi

  echo "$label port $port: stopped."
}

stop_port 8000 "Backend"
stop_port 3000 "Frontend"

rm -f "$PROJECT_ROOT"/logs/*.pid 2>/dev/null || true

echo "============================================================"
echo "Local app stopped."
echo "============================================================"
