#!/bin/bash

# EdukaAI Studio Launcher
# This script starts both backend and frontend

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Starting EdukaAI Studio..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start Backend
echo -e "${BLUE}▶ Starting Backend...${NC}"
cd backend
source .venv/bin/activate
python run.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Frontend
echo -e "${BLUE}▶ Starting Frontend...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}✓ EdukaAI Studio is running!${NC}"
echo ""
echo "📱 Frontend: http://localhost:${VITE_PORT:-3030}"
echo "🔌 Backend API: http://localhost:${EDUKAAI_PORT:-8000}"
echo "📚 API Docs: http://localhost:${EDUKAAI_PORT:-8000}/docs"
echo ""
echo "Environment Variables:"
echo "  Backend: EDUKAAI_PORT=${EDUKAAI_PORT:-8000} (default: 8000)"
echo "  Frontend: VITE_PORT=${VITE_PORT:-3030} (default: 3030)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
