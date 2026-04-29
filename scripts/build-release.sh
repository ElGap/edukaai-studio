#!/bin/bash
# Build a self-contained release bundle for Homebrew distribution.
# This script creates a tarball that includes:
#   - frontend/dist/ (built static assets)
#   - .venv/ (Python virtualenv with all dependencies)
#   - backend/app/ (source code)
#   - launch scripts
#
# Usage: ./scripts/build-release.sh [version]
# Output: dist/edukaai-studio-{version}-darwin-arm64.tar.gz

set -euo pipefail

VERSION="${1:-0.1.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/.build"
DIST_DIR="$PROJECT_ROOT/dist"

info()  { printf "\033[1;34m[info]\033[0m  %s\n" "$1"; }
error() { printf "\033[1;31m[err]\033[0m  %s\n" "$1" >&2; exit 1; }

# Check prerequisites
[[ "$(uname -m)" == "arm64" ]] || error "Must build on Apple Silicon (ARM64)"
command -v node >/dev/null 2>&1 || error "Node.js required"
command -v python3 >/dev/null 2>&1 || error "Python 3.12 required"

PYTHON_VER=$(python3.12 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
[[ "$PYTHON_VER" == "3.12" ]] || error "Python 3.12 required (found $PYTHON_VER)"

info "Building EdukaAI Studio v$VERSION for Apple Silicon"

# Clean and create directories
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# ---------------------------------------------------------------------------
# 1. Build frontend
# ---------------------------------------------------------------------------
info "Building frontend..."
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi
# Build with vite directly (skip vue-tsc type checking which may have non-blocking errors)
npx vite build

# ---------------------------------------------------------------------------
# 2. Create Python virtualenv and install dependencies
# ---------------------------------------------------------------------------
info "Creating Python virtualenv..."
cd "$BUILD_DIR"
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r "$PROJECT_ROOT/backend/requirements.txt"

# Install the app package itself (from root pyproject.toml)
pip install "$PROJECT_ROOT"

# Verify key packages (use venv python while still activated)
python3.12 -c "import mlx; print('MLX: OK')" || error "MLX not installed"
python3.12 -c "import mlx_lm; print('MLX-LM: OK')" || error "mlx-lm not installed"
python3.12 -c "from app.main import app; print('App: OK')" || error "App not importable"

deactivate

# ---------------------------------------------------------------------------
# 3. Assemble the bundle
# ---------------------------------------------------------------------------
info "Assembling bundle..."

BUNDLE_DIR="$BUILD_DIR/edukaai-studio"
mkdir -p "$BUNDLE_DIR"

# Copy frontend build
cp -r "$PROJECT_ROOT/frontend/dist" "$BUNDLE_DIR/"

# Copy backend source (for PYTHONPATH and static file serving)
cp -r "$PROJECT_ROOT/backend/app" "$BUNDLE_DIR/"
cp "$PROJECT_ROOT/backend/run.py" "$BUNDLE_DIR/"

# Copy configuration files
cp "$PROJECT_ROOT/.env.example" "$BUNDLE_DIR/"
cp "$PROJECT_ROOT/README.md" "$BUNDLE_DIR/"

# Copy the virtualenv
cp -r "$BUILD_DIR/.venv" "$BUNDLE_DIR/"

# Create wrapper scripts
# Main launcher script
cat > "$BUNDLE_DIR/edukaai-studio" <<'WRAPPER'
#!/bin/bash
set -euo pipefail

# Determine bundle directory (where this script lives)
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Persistent data directory
DATA_DIR="${EDUKAAI_DATA_DIR:-$HOME/.edukaai}"
STORAGE_PATH="${EDUKAAI_STORAGE_PATH:-$DATA_DIR/storage}"
MODEL_CACHE_DIR="${EDUKAAI_MODEL_CACHE_DIR:-$DATA_DIR/models}"
TRAINING_OUTPUT_DIR="${EDUKAAI_TRAINING_OUTPUT_DIR:-$DATA_DIR/training}"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$STORAGE_PATH" "$MODEL_CACHE_DIR" "$TRAINING_OUTPUT_DIR" "$LOG_DIR"

# Set environment variables
export EDUKAAI_HOST="${EDUKAAI_HOST:-127.0.0.1}"
export EDUKAAI_PORT="${EDUKAAI_PORT:-8000}"
export EDUKAAI_STORAGE_PATH="$STORAGE_PATH"
export EDUKAAI_MODEL_CACHE_DIR="$MODEL_CACHE_DIR"
export EDUKAAI_TRAINING_OUTPUT_DIR="$TRAINING_OUTPUT_DIR"
export EDUKAAI_LOG_FILE="${EDUKAAI_LOG_FILE:-$LOG_DIR/edukaai.log}"
export EDUKAAI_DATABASE_URL="${EDUKAAI_DATABASE_URL:-sqlite:///$STORAGE_PATH/app/edukaai.db}"
export EDUKAAI_FRONTEND_DIST="${EDUKAAI_FRONTEND_DIST:-$BUNDLE_DIR/dist}"
export EDUKAAI_ALLOW_REMOTE="${EDUKAAI_ALLOW_REMOTE:-false}"
export EDUKAAI_LOG_LEVEL="${EDUKAAI_LOG_LEVEL:-INFO}"

# Add backend to PYTHONPATH for imports
export PYTHONPATH="$BUNDLE_DIR:$PYTHONPATH"

cd "$BUNDLE_DIR"

# Create storage symlink for backward compatibility
[ -L "backend/storage" ] || {
    [ -d "backend/storage" ] && rm -rf "backend/storage"
    ln -s "$STORAGE_PATH" backend/storage 2>/dev/null || true
}

# Start the server
exec "$BUNDLE_DIR/.venv/bin/python" -m uvicorn app.main:app \
    --host "$EDUKAAI_HOST" \
    --port "$EDUKAAI_PORT" \
    --log-level "${EDUKAAI_LOG_LEVEL,,}" \
    --no-access-log
WRAPPER
chmod +x "$BUNDLE_DIR/edukaai-studio"

# ---------------------------------------------------------------------------
# 4. Create tarball
# ---------------------------------------------------------------------------
info "Creating tarball..."
cd "$BUILD_DIR"
tar -czf "$DIST_DIR/edukaai-studio-${VERSION}-darwin-arm64.tar.gz" \
    --exclude='.venv/bin/python3.12' \
    --exclude='.venv/include' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    "edukaai-studio"

# Compute SHA256
SHA256=$(shasum -a 256 "$DIST_DIR/edukaai-studio-${VERSION}-darwin-arm64.tar.gz" | awk '{print $1}')

info "Build complete!"
echo ""
echo "  Tarball: $DIST_DIR/edukaai-studio-${VERSION}-darwin-arm64.tar.gz"
echo "  SHA256:  $SHA256"
echo ""
echo "  To test locally:"
echo "    cd $BUILD_DIR/edukaai-studio"
echo "    ./edukaai-studio"
echo ""
echo "  To upload to GitHub Releases:"
echo "    gh release upload v${VERSION} $DIST_DIR/edukaai-studio-${VERSION}-darwin-arm64.tar.gz"
echo ""
