#!/bin/bash

# EdukaAI Studio Launcher
# This script starts both backend and frontend

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Starting EdukaAI Studio..."
echo ""

# Function to get all child PIDs recursively
get_child_pids() {
    local parent_pid=$1
    local children=$(pgrep -P $parent_pid 2>/dev/null)
    echo "$parent_pid"
    for child in $children; do
        get_child_pids $child
    done
}

# Function to cleanup on exit - kills entire process tree
cleanup() {
    echo ""
    echo "Shutting down..."
    
    # Kill backend and all its children
    if [ -n "$BACKEND_PID" ]; then
        for pid in $(get_child_pids $BACKEND_PID); do
            kill $pid 2>/dev/null
        done
    fi
    
    # Kill frontend and all its children
    if [ -n "$FRONTEND_PID" ]; then
        for pid in $(get_child_pids $FRONTEND_PID); do
            kill $pid 2>/dev/null
        done
    fi
    
    # Wait a moment for graceful shutdown
    sleep 1
    
    # Force kill any remaining processes
    if [ -n "$BACKEND_PID" ]; then
        for pid in $(get_child_pids $BACKEND_PID); do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    if [ -n "$FRONTEND_PID" ]; then
        for pid in $(get_child_pids $FRONTEND_PID); do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    exit 0
}

trap cleanup INT TERM

# Start Backend
echo "Starting Backend..."
cd backend
source .venv/bin/activate
python run.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "EdukaAI Studio is running!"
echo ""
echo "Frontend: http://localhost:${VITE_PORT:-3030}"
echo "Backend API: http://localhost:${EDUKAAI_PORT:-8000}"
echo "API Docs: http://localhost:${EDUKAAI_PORT:-8000}/docs"
echo ""
echo "Environment Variables:"
echo "  Backend: EDUKAAI_PORT=${EDUKAAI_PORT:-8000} (default: 8000)"
echo "  Frontend: VITE_PORT=${VITE_PORT:-3030} (default: 3030)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
