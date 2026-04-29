#!/bin/bash
set -euo pipefail

# ============================================================================
# EdukaAI Studio Installer
# https://github.com/elgap/edukaai-studio
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh | bash -s -- --yes
#   ./install.sh                     # developer mode (from repo root)
#   ./install.sh --uninstall
# ============================================================================

REPO_URL="https://github.com/elgap/edukaai-studio"
INSTALL_DIR="${EDUKAAI_INSTALL_DIR:-$HOME/.edukaai/studio}"
AUTO_YES=false
MODE=""
UNINSTALL=false

# ---- Helpers ----

info()  { printf "\033[1;34m[info]\033[0m  %s\n" "$1"; }
warn()  { printf "\033[1;33m[warn]\033[0m  %s\n" "$1"; }
error() { printf "\033[1;31m[error]\033[0m %s\n" "$1" >&2; exit 1; }

read_tty() {
    if [ -t 0 ]; then read "$@"
    else read "$@" < /dev/tty
    fi
}

cmd_exists() { command -v "$1" >/dev/null 2>&1; }

confirm() {
    if [ "$AUTO_YES" = true ]; then return 0; fi
    read_tty -p "$1 (y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

cleanup_on_fail() {
    if [ -n "${CLONED:-}" ] && [ "$MODE" = "user" ] && [ -d "$INSTALL_DIR/.git" ]; then
        warn "Installation failed. Cleaning up $INSTALL_DIR ..."
        rm -rf "$INSTALL_DIR"
    fi
}
trap cleanup_on_fail ERR

# ---- Parse Args ----

for arg in "$@"; do
    case "$arg" in
        --yes|-y)    AUTO_YES=true ;;
        --uninstall) UNINSTALL=true ;;
        --dir=*)     INSTALL_DIR="${arg#--dir=}" ;;
    esac
done

# ---- Uninstall ----

if [ "$UNINSTALL" = true ]; then
    info "Uninstalling EdukaAI Studio..."
    if [ -d "$INSTALL_DIR" ]; then
        confirm "Remove $INSTALL_DIR?"
        rm -rf "$INSTALL_DIR"
        rm -f "$HOME/Desktop/EdukaAI-Studio.command"
        rm -f "$HOME/.local/bin/edukaai-studio"
        info "Uninstalled."
    else
        warn "$INSTALL_DIR not found. Already uninstalled."
    fi
    exit 0
fi

# ---- Detect Mode ----

if [ -d ".git" ] && [ -f "backend/requirements.txt" ] && [ -f "install.sh" ]; then
    MODE="developer"
    INSTALL_DIR="$(pwd)"
else
    MODE="user"
fi

info "EdukaAI Studio Installer  (mode: $MODE)"

# ---- Prerequisites ----

[[ "$OSTYPE" == darwin* ]] || error "macOS is required (Apple Silicon recommended)"

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    warn "Apple Silicon (M1/M2/M3/M4) recommended. You have: $ARCH"
    confirm "Continue anyway?" || exit 0
fi

cmd_exists python3 || error "Python 3.10+ required. Install from https://www.python.org/downloads/"
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ "$(printf '%s\n' "3.10" "$PY_VER" | sort -V | head -n1)" == "3.10" ]] \
    || error "Python 3.10+ required (found $PY_VER)"

cmd_exists node || error "Node.js 18+ required. Install: brew install node  or  https://nodejs.org/"
NODE_VER=$(node -c 'console.log(process.versions.node.split(".")[0])' 2>/dev/null || node -e 'console.log(process.versions.node.split(".")[0])')
[[ "${NODE_VER:-0}" -ge 18 ]] || error "Node.js 18+ required (found v$NODE_VER)"

cmd_exists git || error "Git required. Install Xcode Command Line Tools: xcode-select --install"

info "Prerequisites OK  (Python $PY_VER, Node v$NODE_VER, $ARCH)"

# ---- Confirm ----

if [ "$MODE" = "user" ]; then
    info "Install location: $INSTALL_DIR"
    confirm "Install EdukaAI Studio?" || exit 0
fi

# ---- Clone (user mode) ----

if [ "$MODE" = "user" ]; then
    mkdir -p "$INSTALL_DIR"
    if [ ! -f "$INSTALL_DIR/README.md" ]; then
        info "Downloading EdukaAI Studio..."
        git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
        CLONED=1
        # Verify clone integrity
        cd "$INSTALL_DIR"
        git fetch --depth 1 origin main 2>/dev/null || true
        cd - >/dev/null
    else
        info "Existing installation found. Updating..."
        cd "$INSTALL_DIR"
        git pull --ff-only 2>/dev/null || warn "Could not git pull. Using existing code."
        cd - >/dev/null
    fi
fi

cd "$INSTALL_DIR"

# ---- Backend ----

info "Setting up Python environment..."
cd backend

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd ..

# ---- Frontend ----

info "Setting up Node.js environment..."
cd frontend

if [ ! -d "node_modules" ]; then
    npm install
fi

cd ..

# ---- Storage directories ----

mkdir -p backend/storage/app/logs
mkdir -p backend/storage/app/temp
mkdir -p backend/storage/app/cache
mkdir -p backend/storage/datasets
mkdir -p backend/storage/runs
mkdir -p backend/storage/exports

# ---- .env ----

if [ ! -f .env ]; then
    cp .env.example .env
    info "Created .env from .env.example (review and customize if needed)"
fi

# ---- CLI symlink ----

mkdir -p "$HOME/.local/bin" 2>/dev/null || true
ln -sf "$INSTALL_DIR/launch.sh" "$HOME/.local/bin/edukaai-studio" 2>/dev/null || true

# ---- Desktop shortcut (user mode) ----

if [ "$MODE" = "user" ]; then
    SHORTCUT="$HOME/Desktop/EdukaAI-Studio.command"
    cat > "$SHORTCUT" <<EOF
#!/bin/bash
cd "$INSTALL_DIR"
exec ./launch.sh
EOF
    chmod +x "$SHORTCUT"
    info "Desktop shortcut created"
fi

# ---- Done ----

echo ""
info "Installation complete!"
echo ""
echo "  Start:      ./launch.sh"
echo "  Or:         edukaai-studio          (if ~/.local/bin is in PATH)"
echo "  Desktop:    Double-click EdukaAI-Studio.command"
echo "  Uninstall:  ./install.sh --uninstall"
echo "  Open:       http://localhost:3030"
echo ""

if [ "$MODE" = "user" ]; then
    if confirm "Launch now?"; then
        exec ./launch.sh
    fi
fi
