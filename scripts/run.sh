#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  echo "No .env found — copying .env.example. Fill in RUNPOD_API_KEY before rendering."
  cp .env.example .env
fi

# Install deps if venv doesn't exist
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

source .venv/bin/activate

echo "Starting FastAPI backend on :8000…"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting Gradio frontend on :7860…"
python frontend/app.py &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
echo "SoloStudio-AI running. Open http://localhost:7860"
wait
