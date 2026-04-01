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
    
    # Send SIGTERM (graceful shutdown) to backend first
    if [ -n "$BACKEND_PID" ]; then
        # Try graceful shutdown first
        kill -TERM $BACKEND_PID 2>/dev/null
        # Wait for backend to shut down (gives multiprocessing time to clean up)
        for i in {1..5}; do
            if ! kill -0 $BACKEND_PID 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
    fi
    
    # Kill frontend
    if [ -n "$FRONTEND_PID" ]; then
        kill -TERM $FRONTEND_PID 2>/dev/null
        sleep 0.5
    fi
    
    # Force kill any remaining backend processes
    if [ -n "$BACKEND_PID" ]; then
        for pid in $(get_child_pids $BACKEND_PID); do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    # Force kill any remaining frontend processes  
    if [ -n "$FRONTEND_PID" ]; then
        for pid in $(get_child_pids $FRONTEND_PID); do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    # Give Python's resource_tracker time to clean up semaphores
    sleep 1
    
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
