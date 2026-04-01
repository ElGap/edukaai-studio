#!/bin/bash

# EdukaAI Studio Installer Script
# This script sets up the complete development environment

set -e

echo "🚀 EdukaAI Studio Installer"
echo "=========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS with Apple Silicon
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ Error: This installer only supports macOS${NC}"
    exit 1
fi

if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: EdukaAI Studio is optimized for Apple Silicon (M1/M2/M3)${NC}"
    echo -e "${YELLOW}   It may not work correctly on Intel Macs${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📍 Installation directory: $SCRIPT_DIR"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.10 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | grep -o '3\.[0-9]*' | head -1)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}❌ Python 3.10 or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Please install Node.js 18 or higher from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | grep -o '[0-9]*' | head -1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ Node.js 18 or higher is required (found $(node --version))${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Node.js $(node --version) found${NC}"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}⚠️  Git not found. Some features may not work.${NC}"
fi

echo ""

# Setup Backend
echo "🔧 Setting up Backend..."
cd backend

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo -e "${GREEN}✓ Backend setup complete${NC}"
echo ""

# Setup Frontend
echo "🔧 Setting up Frontend..."
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
else
    echo "Node modules already installed, skipping..."
fi

echo -e "${GREEN}✓ Frontend setup complete${NC}"
echo ""

# Create storage directories
echo "📁 Creating storage directories..."
cd ..
mkdir -p backend/storage/datasets
mkdir -p backend/storage/runs/downloaded_models
mkdir -p backend/storage/db
mkdir -p backend/storage/exports

echo -e "${GREEN}✓ Storage directories created${NC}"
echo ""

# Run tests
echo "🧪 Running tests..."
cd backend
source .venv/bin/activate
python -m pytest tests/ -v --tb=short 2>&1 | head -50
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests failed, but installation can continue${NC}"
fi
cd ..
echo ""

# Create launch script
echo "📝 Creating launch script..."
cat > launch.sh << 'EOF'
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
EOF

chmod +x launch.sh

echo -e "${GREEN}✓ Launch script created${NC}"
echo ""

# Installation complete
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "You can now start EdukaAI Studio by running:"
echo ""
echo "   ./launch.sh"
echo ""
echo "Or manually start components:"
echo ""
echo "   Terminal 1 (Backend):"
echo "   cd backend && source .venv/bin/activate && python run.py"
echo ""
echo "   Terminal 2 (Frontend):"
echo "   cd frontend && npm run dev"
echo ""
echo "Then open: http://localhost:5173"
echo ""
echo "📖 Documentation: README.md"
echo "🐛 Issues: https://github.com/elgap/edukaai-studio/issues"
echo ""
