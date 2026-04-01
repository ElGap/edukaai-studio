#!/bin/bash

# EdukaAI Studio Installer
# Context-aware installation script for both users and developers
# 
# Usage:
#   User (one-line install):  curl -fsSL ... | bash
#   User (with auto-yes):     curl -fsSL ... | bash -s -- --yes
#   Developer:                ./install.sh

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_URL="https://github.com/elgap/edukaai-studio"
AUTO_YES=false

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Detect if running in developer mode (from git repo) or user mode (piped)
detect_mode() {
    if [ -d ".git" ] && [ -f "README.md" ] && [ -f "backend/requirements.txt" ]; then
        echo "developer"
    else
        echo "user"
    fi
}

# Read from TTY when script is piped (for user prompts)
read_tty() {
    if [ -t 0 ]; then
        read "$@"
    else
        read "$@" < /dev/tty
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# PREREQUISITE CHECKS
# ============================================================================

check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo "Error: EdukaAI Studio requires macOS"
        exit 1
    fi
}

check_apple_silicon() {
    local arch=$(uname -m)
    if [[ "$arch" != "arm64" ]]; then
        echo "Warning: This tool is optimized for Apple Silicon Macs"
        echo "   (M1/M2/M3/M4 chips). It may not work correctly on Intel Macs."
        echo ""
        
        if [ "$AUTO_YES" = false ]; then
            read_tty -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            echo "Auto-yes enabled, continuing..."
        fi
    fi
    
    echo "Apple Silicon detected"
}

check_macos_version() {
    local os_version=$(sw_vers -productVersion)
    local major_version=$(echo "$os_version" | cut -d. -f1)
    
    if [[ "$major_version" -lt 12 ]]; then
        echo "Warning: macOS 12.0+ recommended (you have $os_version)"
    fi
}

check_python() {
    echo ""
    echo "Checking Python..."
    
    if ! command_exists python3; then
        echo "Error: Python 3 is not installed"
        echo ""
        echo "To install Python:"
        echo "   1. Visit: https://www.python.org/downloads/"
        echo "   2. Download Python 3.10 or higher"
        echo "   3. Run the installer"
        echo ""
        echo "After installing Python, run this script again."
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | grep -o '3\.[0-9]*' | head -1)
    local required_version="3.10"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        echo "Error: Python 3.10 or higher is required (found $python_version)"
        echo ""
        echo "To install Python:"
        echo "   1. Visit: https://www.python.org/downloads/"
        echo "   2. Download Python 3.10 or higher"
        echo "   3. Run the installer"
        echo ""
        echo "After installing Python, run this script again."
        exit 1
    fi
    
    echo "Python $python_version found"
}

check_nodejs() {
    echo ""
    echo "Checking Node.js..."
    
    if ! command_exists node; then
        if [ "$MODE" = "user" ]; then
            echo "Node.js not found. It will be installed automatically."
            INSTALL_NODE=true
        else
            echo "Error: Node.js is not installed"
            echo "Please install Node.js 18 or higher from https://nodejs.org/"
            exit 1
        fi
    else
        local node_version=$(node --version | grep -o 'v[0-9]*' | head -1 | tr -d 'v')
        if [ "$node_version" -lt 18 ]; then
            if [ "$MODE" = "user" ]; then
                echo "Node.js version $node_version found, but 18+ is required"
                echo "   Node.js will be upgraded automatically during installation."
                INSTALL_NODE=true
            else
                echo "Error: Node.js 18 or higher is required (found $(node --version))"
                exit 1
            fi
        else
            echo "Node.js $(node --version) found"
        fi
    fi
}

check_git() {
    echo ""
    echo "Checking Git..."
    
    if ! command_exists git; then
        echo "Warning: Git not found. Installing Xcode Command Line Tools..."
        xcode-select --install 2>/dev/null || true
        echo "   Please complete the installation and run this script again."
        exit 1
    fi
    
    echo "Git found"
}

# ============================================================================
# INSTALLATION STEPS
# ============================================================================

setup_installation_directory() {
    # Get the directory where the script was invoked from
    local invoke_dir="$(pwd)"
    
    if [ "$MODE" = "user" ]; then
        echo ""
        echo "Setting up installation directory..."
        
        # Install in current directory (where user ran the command)
        INSTALL_DIR="${invoke_dir}/edukaai-studio"
        
        if [ -d "$INSTALL_DIR" ]; then
            echo "   Existing installation found at: $INSTALL_DIR"
            echo "   Updating..."
            rm -rf "$INSTALL_DIR"
        fi
        
        mkdir -p "$INSTALL_DIR"
        cd "$INSTALL_DIR"
        
        # Download repository
        echo ""
        echo "Downloading EdukaAI Studio..."
        echo "   Installing to: $INSTALL_DIR"
        git clone --depth 1 "$REPO_URL.git" . 2>&1 | grep -v "Receiving objects" || true
        
        if [ ! -f "README.md" ]; then
            echo "Error: Failed to download EdukaAI Studio"
            exit 1
        fi
        
        echo "Downloaded successfully"
    else
        # Developer mode - already in repo
        INSTALL_DIR="$(pwd)"
        echo "Developer mode detected. Installing in: $INSTALL_DIR"
    fi
}

install_nodejs() {
    if [ "$INSTALL_NODE" = true ]; then
        echo ""
        echo "Installing Node.js..."
        echo "   Downloading Node.js installer..."
        
        local node_installer="node-installer.pkg"
        curl -fsSL "https://nodejs.org/dist/v20.11.0/node-v20.11.0.pkg" -o "$node_installer" || {
            echo "Error: Failed to download Node.js"
            echo "   Please install Node.js manually from https://nodejs.org/"
            exit 1
        }
        
        echo "   Installing Node.js (you may need to enter your password)..."
        sudo installer -pkg "$node_installer" -target / || {
            echo "Error: Failed to install Node.js"
            exit 1
        }
        rm "$node_installer"
        
        export PATH="/usr/local/bin:$PATH"
        echo "Node.js installed"
    fi
}

setup_backend() {
    echo ""
    echo "Setting up Backend..."
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
    
    echo "Backend setup complete"
    cd ..
}

setup_frontend() {
    echo ""
    echo "Setting up Frontend..."
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install
    else
        echo "Node modules already installed, skipping..."
    fi
    
    echo "Frontend setup complete"
    cd ..
}

create_directories() {
    echo ""
    echo "Creating storage directories..."
    
    mkdir -p backend/storage/app/logs
    mkdir -p backend/storage/app/temp
    mkdir -p backend/storage/app/cache
    mkdir -p backend/storage/datasets
    mkdir -p backend/storage/runs
    mkdir -p backend/storage/exports
    
    # Create .gitkeep files for empty directory tracking
    touch backend/storage/datasets/.gitkeep 2>/dev/null || true
    touch backend/storage/runs/.gitkeep 2>/dev/null || true
    touch backend/storage/exports/.gitkeep 2>/dev/null || true
    touch backend/storage/app/.gitkeep 2>/dev/null || true
    
    echo "Storage directories created"
}

run_tests() {
    if [ "$MODE" = "developer" ]; then
        echo ""
        echo "Running tests..."
        cd backend
        source .venv/bin/activate
        python -m pytest tests/ -v --tb=short 2>&1 | head -50
        local test_result=${PIPESTATUS[0]}
        cd ..
        
        if [ $test_result -eq 0 ]; then
            echo "Tests passed"
        else
            echo "Some tests failed, but installation can continue"
        fi
    fi
}

create_desktop_shortcut() {
    if [ "$MODE" = "user" ]; then
        echo ""
        echo "Creating Desktop shortcut..."
        
        local desktop_link="${HOME}/Desktop/EdukaAI-Studio.command"
        cat > "$desktop_link" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
./launch.sh
EOF
        chmod +x "$desktop_link"
        echo "Created Desktop shortcut: ~/Desktop/EdukaAI-Studio.command"
    fi
}

# ============================================================================
# MAIN INSTALLATION FLOW
# ============================================================================

main() {
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --yes|-y)
                AUTO_YES=true
                shift
                ;;
        esac
    done
    
    # Detect mode
    MODE=$(detect_mode)
    
    # Header
    echo ""
    echo "EdukaAI Studio Installer"
    echo "Mode: $MODE"
    echo ""
    
    # Welcome / Continue prompt (user mode only, or if interactive)
    if [ "$MODE" = "user" ]; then
        echo "Welcome! This installer will set up EdukaAI Studio on your Mac."
        echo ""
        echo "What will be installed:"
        echo "   - Python environment"
        echo "   - EdukaAI Studio application"
        echo "   - Required AI/ML libraries"
        echo ""
        echo "Installation location: $(pwd)/edukaai-studio"
        echo ""
        
        if [ "$AUTO_YES" = false ]; then
            read_tty -p "Continue with installation? (Y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]] && [ -n "$REPLY" ]; then
                echo "Installation cancelled."
                exit 0
            fi
        fi
    fi
    
    # System checks
    echo ""
    echo "Checking your system..."
    check_macos
    check_apple_silicon
    check_macos_version
    check_python
    check_nodejs
    check_git
    
    # Setup
    setup_installation_directory
    install_nodejs
    setup_backend
    setup_frontend
    create_directories
    run_tests
    create_desktop_shortcut
    
    # Completion
    echo ""
    echo "==================================================="
    echo "Installation Complete!"
    echo "==================================================="
    echo ""
    echo "EdukaAI Studio is ready to use!"
    echo ""
    
    if [ "$MODE" = "user" ]; then
        echo "Installation location: ${INSTALL_DIR}"
        echo ""
        echo "To start the application:"
        echo "   - Double-click 'EdukaAI-Studio.command' on your Desktop"
        echo "   - Or run: ${INSTALL_DIR}/launch.sh"
        echo "   - Or cd ${INSTALL_DIR} && ./launch.sh"
        echo ""
        echo "Getting Started:"
        echo "   1. Open your browser to: http://localhost:3030"
        echo "   2. Upload your training dataset"
        echo "   3. Select a base model and training preset"
        echo "   4. Click 'Start Training'"
        echo ""
        echo "Web page: https://eduka.elgap.ai"
        echo "Documentation: ${INSTALL_DIR}/README.md"
        echo "Support: https://github.com/elgap/edukaai-studio/issues"
        echo ""
        
        # Offer to launch
        if [ "$AUTO_YES" = false ]; then
            read_tty -p "Launch EdukaAI Studio now? (Y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [ -z "$REPLY" ]; then
                echo ""
                echo "Launching EdukaAI Studio..."
                sleep 2
                ./launch.sh
            fi
        else
            echo ""
            echo "Auto-launching EdukaAI Studio..."
            sleep 2
            ./launch.sh
        fi
    else
        # Developer mode
        echo "Installation complete in: $INSTALL_DIR"
        echo ""
        echo "To start the application:"
        echo "   ./launch.sh"
        echo ""
        echo "Or manually:"
        echo "   Terminal 1: cd backend && source .venv/bin/activate && python run.py"
        echo "   Terminal 2: cd frontend && npm run dev"
        echo ""
        echo "Then open: http://localhost:3030"
    fi
}

# Run main function
main "$@"
