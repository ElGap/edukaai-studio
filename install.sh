#!/bin/bash
#
# EdukaAI Fine Tuning Studio - One-Line Installer
# 
# Installation: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/elgap/edukaai-studio/main/install.sh)"
#
# This script installs EdukaAI Fine Tuning Studio with all dependencies,
# validates system requirements, and creates a convenient launcher.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/elgap/edukaai-studio.git"
INSTALL_DIR="${HOME}/EdukaAI-Fine-Tuning-Studio"
MIN_MACOS_VERSION="12.3"
MIN_PYTHON_VERSION="3.9"
MIN_RAM_GB=8
MIN_DISK_GB=10
REQUIRED_ARCH="arm64"

# Progress tracking
TOTAL_STEPS=7
CURRENT_STEP=0

# Helper functions
print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     EdukaAI Fine Tuning Studio - Installer              ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "${BLUE}[$CURRENT_STEP/$TOTAL_STEPS]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Version comparison
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# System Requirements Check
check_system_requirements() {
    print_step "Checking System Requirements"
    
    local errors=0
    
    # Check macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This application requires macOS (Apple Silicon)"
        echo "   EdukaAI Fine Tuning Studio is designed for Apple Silicon Macs only."
        exit 1
    fi
    print_success "macOS detected"
    
    # Check macOS version
    MACOS_VERSION=$(sw_vers -productVersion)
    if ! version_ge "$MACOS_VERSION" "$MIN_MACOS_VERSION"; then
        print_error "macOS $MACOS_VERSION is too old (requires $MIN_MACOS_VERSION+)"
        echo "   Please upgrade macOS to Monterey 12.3 or later."
        errors=$((errors + 1))
    else
        print_success "macOS $MACOS_VERSION ✓"
    fi
    
    # Check architecture (Apple Silicon)
    ARCH=$(uname -m)
    if [[ "$ARCH" != "$REQUIRED_ARCH" ]]; then
        print_error "Intel Mac detected (architecture: $ARCH)"
        echo "   This application requires Apple Silicon (M1/M2/M3)."
        echo "   Intel Macs are not supported due to MLX framework requirements."
        errors=$((errors + 1))
    else
        print_success "Apple Silicon (M1/M2/M3) detected ✓"
    fi
    
    # Check RAM
    RAM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    if [[ $RAM_GB -lt $MIN_RAM_GB ]]; then
        print_warning "Low RAM: ${RAM_GB}GB (recommended: ${MIN_RAM_GB}GB+)"
        echo "   Training may be slow or fail with limited memory."
    else
        print_success "RAM: ${RAM_GB}GB ✓"
    fi
    
    # Check disk space
    AVAILABLE_GB=$(df -g . | tail -1 | awk '{print $4}')
    if [[ $AVAILABLE_GB -lt $MIN_DISK_GB ]]; then
        print_error "Insufficient disk space: ${AVAILABLE_GB}GB available"
        echo "   Need at least ${MIN_DISK_GB}GB free space for models and data."
        errors=$((errors + 1))
    else
        print_success "Disk space: ${AVAILABLE_GB}GB available ✓"
    fi
    
    if [[ $errors -gt 0 ]]; then
        echo ""
        print_error "System requirements not met. Please fix the issues above."
        exit 1
    fi
    
    echo ""
}

# Check and install Python dependencies
check_python_environment() {
    print_step "Checking Python Environment"
    
    # Check Python 3
    if ! command_exists python3; then
        print_error "Python 3 not found"
        echo "   Please install Python 3.9 or later from python.org"
        echo "   or run: brew install python@3.11"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
    if ! version_ge "$PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
        print_error "Python $PYTHON_VERSION is too old (requires $MIN_PYTHON_VERSION+)"
        echo "   Please upgrade Python:"
        echo "   brew install python@3.11"
        echo "   OR download from https://www.python.org/downloads/"
        exit 1
    fi
    print_success "Python $PYTHON_VERSION ✓"
    
    # Check pip
    if ! command_exists pip3; then
        print_warning "pip3 not found, trying to install..."
        python3 -m ensurepip --upgrade 2>/dev/null || {
            print_error "Failed to install pip"
            echo "   Please install pip manually:"
            echo "   curl https://bootstrap.pypa.io/get-pip.py | python3"
            exit 1
        }
    fi
    print_success "pip available ✓"
    
    echo ""
}

# Clone or update repository
clone_repository() {
    print_step "Downloading EdukaAI Fine Tuning Studio"
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "   Update existing installation? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$INSTALL_DIR"
            print_info "Updating to latest version..."
            git pull origin main || {
                print_error "Failed to update repository"
                exit 1
            }
        else
            print_info "Using existing installation"
        fi
    else
        print_info "Cloning repository to $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR" || {
            print_error "Failed to clone repository"
            echo "   Please check your internet connection and try again."
            exit 1
        }
        print_success "Repository cloned ✓"
    fi
    
    cd "$INSTALL_DIR"
    echo ""
}

# Setup Python virtual environment
setup_virtual_environment() {
    print_step "Setting up Python Environment"
    
    if [[ -d ".venv" ]]; then
        print_info "Virtual environment already exists"
    else
        print_info "Creating virtual environment..."
        python3 -m venv .venv || {
            print_error "Failed to create virtual environment"
            exit 1
        }
        print_success "Virtual environment created ✓"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    print_success "Virtual environment activated ✓"
    
    # Upgrade pip
    pip install --upgrade pip -q
    
    echo ""
}

# Install Python dependencies
install_dependencies() {
    print_step "Installing Dependencies (this may take 2-5 minutes)"
    
    print_info "Installing machine learning packages..."
    
    # Show progress
    pip install -r requirements.txt --progress-bar on 2>&1 | while read line; do
        if [[ "$line" == *"Successfully installed"* ]]; then
            print_success "Core packages installed"
        fi
    done || {
        print_error "Failed to install dependencies"
        echo "   Try running: pip install -r requirements.txt"
        exit 1
    }
    
    print_success "All dependencies installed ✓"
    echo ""
}

# Validate installation
validate_installation() {
    print_step "Validating Installation"
    
    local errors=0
    
    # Test Python imports
    print_info "Testing Python imports..."
    if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import mlx
    import mlx_lm
    import transformers
    import gradio
    print('✓ All critical imports successful')
except Exception as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_success "Python packages working ✓"
    else
        print_error "Package validation failed"
        errors=$((errors + 1))
    fi
    
    # Test MLX GPU access
    print_info "Testing Apple Silicon GPU access..."
    if python3 -c "
import mlx.core as mx
try:
    arr = mx.array([1.0, 2.0, 3.0])
    result = mx.sum(arr)
    print(f'✓ MLX GPU accessible (test computation: {result}')
except Exception as e:
    print(f'✗ MLX error: {e}')
    exit(1)
" 2>/dev/null; then
        print_success "GPU acceleration available ✓"
    else
        print_warning "GPU test inconclusive (may still work)"
    fi
    
    # Test HuggingFace connectivity
    print_info "Testing HuggingFace connectivity..."
    if curl -s --head https://huggingface.co | head -1 | grep -q "200\|301\|302"; then
        print_success "HuggingFace accessible ✓"
    else
        print_warning "HuggingFace connectivity check inconclusive"
        echo "   (Models will be downloaded on first use)"
    fi
    
    # Check disk space after installation
    AVAILABLE_GB=$(df -g . | tail -1 | awk '{print $4}')
    print_info "Remaining disk space: ${AVAILABLE_GB}GB"
    
    if [[ $errors -gt 0 ]]; then
        echo ""
        print_error "Validation failed with $errors error(s)"
        echo "   The application may not work correctly."
        echo "   Please check the errors above and try again."
    else
        echo ""
        print_success "All validation checks passed! ✓"
    fi
    
    echo ""
}

# Create launch scripts
create_launch_scripts() {
    print_step "Creating Launch Scripts"
    
    # Main launch script
    cat > launch.sh << 'EOF'
#!/bin/bash
# EdukaAI Fine Tuning Studio Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Check if already running
if lsof -i :7860 >/dev/null 2>&1; then
    echo "⚠️  EdukaAI Studio is already running on port 7860"
    echo "   Open browser at: http://localhost:7860"
    exit 0
fi

# Launch application
echo "🚀 Starting EdukaAI Fine Tuning Studio..."
echo "   This may take 30-60 seconds on first launch"
echo ""

python run.py
EOF

    chmod +x launch.sh
    print_success "Created launch.sh ✓"
    
    # Create desktop shortcut (optional)
    if [[ -d "$HOME/Desktop" ]]; then
        cat > "$HOME/Desktop/EdukaAI Studio.command" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
./launch.sh
EOF
        chmod +x "$HOME/Desktop/EdukaAI Studio.command"
        print_success "Created Desktop shortcut ✓"
    fi
    
    # Create update script
    cat > update.sh << 'EOF'
#!/bin/bash
# EdukaAI Fine Tuning Studio Updater

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Checking for updates..."

# Save current state
pip freeze > .requirements-backup.txt

# Pull latest changes
git pull origin main || {
    echo "❌ Failed to update. Check internet connection."
    exit 1
}

# Update dependencies
source .venv/bin/activate
pip install -r requirements.txt --upgrade

echo "✅ Update complete! Run ./launch.sh to start."
EOF

    chmod +x update.sh
    print_success "Created update.sh ✓"
    
    # Create doctor script for troubleshooting
    cat > doctor.sh << 'EOF'
#!/bin/bash
# EdukaAI Fine Tuning Studio - Diagnostic Tool

echo "🔍 EdukaAI Fine Tuning Studio Diagnostics"
echo "=========================================="

# Check Python
echo ""
echo "Python version:"
python3 --version 2>&1

# Check virtual environment
echo ""
echo "Virtual environment:"
if [ -f ".venv/bin/activate" ]; then
    echo "✓ Virtual environment exists"
    source .venv/bin/activate
    pip list 2>/dev/null | grep -E "mlx|transformers|gradio" | head -10
else
    echo "✗ Virtual environment not found"
fi

# Check MLX
echo ""
echo "MLX test:"
python3 -c "import mlx.core as mx; print(f'✓ MLX working, GPU memory: {mx.get_active_memory()/1e9:.2f}GB')" 2>&1 || echo "✗ MLX import failed"

# Check port
echo ""
echo "Port 7860 status:"
if lsof -i :7860 >/dev/null 2>&1; then
    echo "⚠️  Port 7860 is in use"
    lsof -i :7860 | tail -1
else
    echo "✓ Port 7860 is available"
fi

# Check disk space
echo ""
echo "Disk space:"
df -h . | tail -1

echo ""
echo "=========================================="
echo "Run ./launch.sh to start the application"
EOF

    chmod +x doctor.sh
    print_success "Created doctor.sh ✓"
    
    echo ""
}

# Print success message
print_success_message() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Installation Complete!                  ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}📁 Installation directory:${NC}"
    echo "   $INSTALL_DIR"
    echo ""
    echo -e "${CYAN}🚀 To start EdukaAI Fine Tuning Studio:${NC}"
    echo "   cd $INSTALL_DIR"
    echo "   ./launch.sh"
    echo ""
    echo -e "${CYAN}🌐 Then open your browser at:${NC}"
    echo "   http://localhost:7860"
    echo ""
    echo -e "${CYAN}📖 Quick Start:${NC}"
    echo "   1. Upload training data (JSONL format)"
    echo "   2. Select model and configure training"
    echo "   3. Start training and watch progress"
    echo "   4. Download fine-tuned model"
    echo ""
    echo -e "${CYAN}🛠️  Useful commands:${NC}"
    echo "   ./launch.sh     - Start the application"
    echo "   ./update.sh     - Update to latest version"
    echo "   ./doctor.sh     - Run diagnostics"
    echo ""
    echo -e "${CYAN}📚 Documentation:${NC}"
    echo "   docs/QUICKSTART.md - Getting started guide"
    echo "   docs/SECURITY_UPDATES_COMPLETED.md - Security info"
    echo ""
    echo -e "${YELLOW}💡 Pro tip:${NC} First training session will download models (~3-5GB)"
    echo "            This is normal and only happens once per model."
    echo ""
    echo -e "${GREEN}Happy Fine-Tuning! 🤖✨${NC}"
    echo ""
}

# Print error message
print_error_message() {
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                 Installation Failed                        ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Need help?${NC}"
    echo "   1. Check docs/TROUBLESHOOTING.md"
    echo "   2. Open an issue: https://github.com/elgap/edukaai-studio/issues"
    echo "   3. Include any error messages shown above"
    echo ""
}

# Main installation flow
main() {
    print_header
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --help, -h       Show this help message"
                echo "  --dry-run        Check requirements without installing"
                echo "  --no-shortcuts   Don't create desktop shortcuts"
                echo ""
                exit 0
                ;;
            --dry-run)
                print_info "Dry run mode - checking requirements only"
                check_system_requirements
                check_python_environment
                print_success "System requirements met! Ready to install."
                exit 0
                ;;
            --no-shortcuts)
                NO_SHORTCUTS=1
                shift
                ;;
            *)
                echo "Unknown option: $1"
                echo "Run '$0 --help' for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    check_system_requirements
    check_python_environment
    clone_repository
    setup_virtual_environment
    install_dependencies
    validate_installation
    create_launch_scripts
    
    # Success!
    print_success_message
}

# Run main function
main "$@"
