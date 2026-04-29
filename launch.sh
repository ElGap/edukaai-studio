#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
[ -f .env ] && set -a && source .env && set +a

PORT="${EDUKAAI_PORT:-8000}"
VPORT="${VITE_PORT:-3030}"
PIDFILE="$SCRIPT_DIR/.edukaai.pid"
MAX_WAIT=15

# --- Helpers ---

info()  { printf "\033[1;34m[info]\033[0m  %s\n" "$1"; }
warn()  { printf "\033[1;33m[warn]\033[0m  %s\n" "$1"; }
error() { printf "\033[1;31m[err]\033[0m  %s\n" "$1" >&2; }

cleanup() {
    echo ""
    info "Shutting down..."

    if [ -n "${BACKEND_PID:-}" ]; then
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi

    sleep 1

    # Force kill if still running
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        warn "Backend did not exit gracefully, force killing"
        kill -9 "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        warn "Frontend did not exit gracefully, force killing"
        kill -9 "$FRONTEND_PID" 2>/dev/null || true
    fi

    rm -f "$PIDFILE"
    wait 2>/dev/null
    exit 0
}
trap cleanup INT TERM

check_port() {
    if lsof -i ":$1" -sTCP:LISTEN >/dev/null 2>&1; then
        error "Port $1 is already in use. Stop the existing process or change the port."
        lsof -i ":$1" -sTCP:LISTEN 2>/dev/null | head -5
        exit 1
    fi
}

wait_for_backend() {
    local waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

# --- Detect mode ---

# Single-server mode: if frontend/dist exists, the backend serves static files directly.
# This is the production / Homebrew mode.
# Dual-server mode (development): if frontend/dist does NOT exist, start Vite dev server.
if [ -d "frontend/dist" ]; then
    SINGLE_SERVER=true
    info "Production mode: backend serves frontend static files on :$PORT"
else
    SINGLE_SERVER=false
    info "Development mode: starting separate frontend dev server on :$VPORT"
fi

# --- Pre-flight checks ---

# Check for already-running instance
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        error "EdukaAI Studio is already running (PID $OLD_PID). Stop it first."
        exit 1
    fi
    rm -f "$PIDFILE"
fi

# Check ports
check_port "$PORT"
if [ "$SINGLE_SERVER" = false ]; then
    check_port "$VPORT"
fi

# Check backend venv
if [ ! -d "backend/.venv" ]; then
    error "Backend virtual environment not found. Run: cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check frontend node_modules (only in dev mode)
if [ "$SINGLE_SERVER" = false ]; then
    if [ ! -d "frontend/node_modules" ]; then
        error "Frontend dependencies not installed. Run: cd frontend && npm install"
        exit 1
    fi

    # Check Node.js (only in dev mode)
    if ! command -v node >/dev/null 2>&1; then
        error "Node.js not found. Install Node.js 18+ first."
        exit 1
    fi
fi

# Warn about default secret key
if [ -z "${EDUKAAI_SECRET_KEY:-}" ]; then
    warn "Using default secret key. Set EDUKAAI_SECRET_KEY in .env for security."
fi

# --- Start Backend ---

info "Starting backend on :$PORT ..."
cd backend
source .venv/bin/activate
python run.py &
BACKEND_PID=$!
cd ..

# Write PID file (store backend PID as primary)
echo "$BACKEND_PID" > "$PIDFILE"

info "Waiting for backend to become healthy ..."
if ! wait_for_backend; then
    error "Backend failed to start within ${MAX_WAIT}s. Check logs above."
    kill -TERM "$BACKEND_PID" 2>/dev/null || true
    rm -f "$PIDFILE"
    exit 1
fi
info "Backend is healthy"

# --- Start Frontend (dev mode only) ---

if [ "$SINGLE_SERVER" = false ]; then
    info "Starting frontend dev server on :$VPORT ..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
fi

# --- Ready ---

echo ""
info "EdukaAI Studio is running!"
echo ""
if [ "$SINGLE_SERVER" = true ]; then
    echo "  URL:       http://localhost:$PORT"
    echo "  API Docs:  http://localhost:$PORT/docs"
else
    echo "  Frontend:  http://localhost:$VPORT"
    echo "  Backend:   http://localhost:$PORT"
    echo "  API Docs:  http://localhost:$PORT/docs"
fi
echo ""
echo "  Press Ctrl+C to stop"
echo ""

wait
