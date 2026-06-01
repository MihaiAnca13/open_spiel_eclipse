#!/bin/bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$DIR/../.." && pwd )"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
FRONTEND_CMD="${FRONTEND_CMD:-}"
WSL_BUN_BIN="${WSL_BUN_BIN:-$HOME/.bun/bin/bun}"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/build/python${PYTHONPATH:+:$PYTHONPATH}"

if [ -s "$NVM_DIR/nvm.sh" ]; then
    # Align non-interactive runs with the user's interactive WSL shell.
    . "$NVM_DIR/nvm.sh"
fi

echo "=========================================="
echo "Starting Eclipse Digital Services"
echo "=========================================="

# 1. Start FastAPI Backend
echo "-> Launching FastAPI Backend on port 8000..."
cd "$DIR"
"$PYTHON_BIN" -m uvicorn api.main:app --reload --app-dir "$DIR" &
BACKEND_PID=$!

# 2. Start Visualizer Frontend
echo "-> Launching React TS Visualizer..."
cd "$DIR/visualizer" || exit
if [ -n "$FRONTEND_CMD" ]; then
    $FRONTEND_CMD &
elif [ -x "$WSL_BUN_BIN" ]; then
    "$WSL_BUN_BIN" run dev &
elif command -v bun >/dev/null 2>&1; then
    bun run dev &
else
    npm run dev &
fi
FRONTEND_PID=$!

# Trap Ctrl+C (SIGINT) and kill both background processes
cleanup() {
    echo ""
    echo "=========================================="
    echo "Shutting down servers gracefully..."
    echo "=========================================="
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep running to monitor processes and stream logs
wait
