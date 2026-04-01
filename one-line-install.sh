#!/bin/bash

# EdukaAI Studio - One-Line Installer
# For Apple Silicon users who want to fine-tune LLMs without being developers
# Usage: curl -fsSL ... | bash
#        curl -fsSL ... | bash -s -- --yes

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTALL_DIR="${HOME}/Applications/EdukaAI-Studio"
REPO_URL="https://github.com/elgap/edukaai-studio"
VERSION="latest"
AUTO_YES=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        *)
            VERSION="$arg"
            ;;
    esac
done

# Helper function to read from TTY (handles piped execution)
read_tty() {
    if [ -t 0 ]; then
        # stdin is a terminal, read normally
        read "$@"
    else
        # piped execution, read from TTY device
        read "$@" < /dev/tty
    fi
}

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          EdukaAI Studio Installer                          ║"
echo "║     Fine-tune LLMs on Your Apple Silicon Mac               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Welcome message
echo "Welcome! This installer will set up EdukaAI Studio on your Mac."
echo ""
echo "📋 What will be installed:"
echo "   • Python environment (if needed)"
echo "   • EdukaAI Studio application"
echo "   • Required AI/ML libraries"
echo ""
echo "💾 Installation location: ${INSTALL_DIR}"
echo ""
echo "⚠️  Requirements:"
echo "   • macOS with Apple Silicon (M1/M2/M3/M4)"
echo "   • macOS 12.0 or later"
echo "   • Internet connection"
echo "   • ~2GB free disk space"
echo ""
if [ "$AUTO_YES" = true ]; then
    echo "ℹ️  Running in auto mode (--yes flag detected). All prompts will be skipped."
    echo ""
fi

if [ "$AUTO_YES" = false ]; then
    read_tty -p "Continue with installation? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [ -n "$REPLY" ]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# Check system requirements
echo ""
echo "🔍 Checking your system..."
echo ""

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ Error: EdukaAI Studio requires macOS${NC}"
    exit 1
fi

# Check Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: This tool is optimized for Apple Silicon Macs${NC}"
    echo -e "${YELLOW}   (M1/M2/M3/M4 chips). It may not work correctly on Intel Macs.${NC}"
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

echo -e "${GREEN}✓ Apple Silicon detected${NC}"

# Check macOS version (need 12.0+)
OS_VERSION=$(sw_vers -productVersion)
MAJOR_VERSION=$(echo "$OS_VERSION" | cut -d. -f1)
if [[ "$MAJOR_VERSION" -lt 12 ]]; then
    echo -e "${YELLOW}⚠️  Warning: macOS 12.0+ recommended (you have $OS_VERSION)${NC}"
fi

# Check for Python
echo ""
echo "🐍 Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo ""
    echo "📥 To install Python:"
    echo "   1. Visit: https://www.python.org/downloads/"
    echo "   2. Download Python 3.10 or higher"
    echo "   3. Run the installer"
    echo ""
    echo "After installing Python, run this script again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -o '3\.[0-9]*' | head -1)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.10 or higher is required (found $PYTHON_VERSION)${NC}"
    echo ""
    echo "📥 To install Python:"
    echo "   1. Visit: https://www.python.org/downloads/"
    echo "   2. Download Python 3.10 or higher"
    echo "   3. Run the installer"
    echo ""
    echo "After installing Python, run this script again."
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Node.js
echo ""
echo "📦 Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js not found. It will be installed automatically.${NC}"
    INSTALL_NODE=true
else
    NODE_VERSION=$(node --version | grep -o 'v[0-9]*' | head -1 | tr -d 'v')
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo -e "${YELLOW}⚠️  Node.js version $NODE_VERSION found, but 18+ is required${NC}"
        echo "   Node.js will be upgraded automatically during installation."
        INSTALL_NODE=true
    else
        echo -e "${GREEN}✓ Node.js $(node --version) found${NC}"
    fi
fi

# Check Git
echo ""
echo "📥 Checking Git..."
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}⚠️  Git not found. Installing Xcode Command Line Tools...${NC}"
    xcode-select --install 2>/dev/null || true
    echo "   Please complete the installation and run this script again."
    exit 1
fi
echo -e "${GREEN}✓ Git found${NC}"

# Create installation directory
echo ""
echo "📁 Setting up installation directory..."
if [ -d "$INSTALL_DIR" ]; then
    echo "   Existing installation found. Updating..."
    rm -rf "$INSTALL_DIR"
fi
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download/clone the repository
echo ""
echo "⬇️  Downloading EdukaAI Studio..."
if [[ "$VERSION" == "latest" ]]; then
    echo "   Fetching latest version..."
    git clone --depth 1 "$REPO_URL.git" . 2>&1 | grep -v "Receiving objects" || true
else
    echo "   Fetching version $VERSION..."
    git clone --depth 1 --branch "$VERSION" "$REPO_URL.git" . 2>&1 | grep -v "Receiving objects" || true
fi

if [ ! -f "README.md" ]; then
    echo -e "${RED}❌ Failed to download EdukaAI Studio${NC}"
    echo "   Please check your internet connection and try again."
    exit 1
fi

echo -e "${GREEN}✓ Downloaded successfully${NC}"

# Create .gitkeep files for storage directories
echo ""
echo "📁 Setting up storage directories..."
touch backend/storage/datasets/.gitkeep 2>/dev/null || true
touch backend/storage/runs/.gitkeep 2>/dev/null || true
touch backend/storage/exports/.gitkeep 2>/dev/null || true
touch backend/storage/app/.gitkeep 2>/dev/null || true
touch backend/test_data/.gitkeep 2>/dev/null || true
echo -e "${GREEN}✓ Storage directories configured${NC}"

# Install Node.js if needed
if [ "$INSTALL_NODE" = true ]; then
    echo ""
    echo "📦 Installing Node.js..."
    echo "   Downloading Node.js installer..."
    
    # Download and install Node.js
    NODE_INSTALLER="node-installer.pkg"
    curl -fsSL "https://nodejs.org/dist/v20.11.0/node-v20.11.0.pkg" -o "$NODE_INSTALLER" || {
        echo -e "${RED}❌ Failed to download Node.js${NC}"
        echo "   Please install Node.js manually from https://nodejs.org/"
        exit 1
    }
    
    echo "   Installing Node.js (you may need to enter your password)..."
    sudo installer -pkg "$NODE_INSTALLER" -target / || {
        echo -e "${RED}❌ Failed to install Node.js${NC}"
        exit 1
    }
    rm "$NODE_INSTALLER"
    
    # Reload shell to get new node
    export PATH="/usr/local/bin:$PATH"
    echo -e "${GREEN}✓ Node.js installed${NC}"
fi

# Run the main installer
echo ""
echo "⚙️  Installing EdukaAI Studio components..."
echo "   This may take 5-10 minutes depending on your internet speed..."
echo ""

./install.sh 2>&1 | tee install.log | while read line; do
    # Show progress indicators
    if echo "$line" | grep -q "Installing"; then
        echo "   $line"
    fi
done

# Check if installation succeeded
if [ ! -f "launch.sh" ]; then
    echo ""
    echo -e "${RED}❌ Installation failed${NC}"
    echo "   Check install.log for details."
    echo "   Common issues:"
    echo "   • Internet connection problems"
    echo "   • Not enough disk space"
    echo "   • Permission denied (try running with sudo)"
    exit 1
fi

# Create Desktop shortcut
echo ""
echo "🖥️  Creating Desktop shortcut..."
DESKTOP_LINK="${HOME}/Desktop/EdukaAI-Studio.command"
cat > "$DESKTOP_LINK" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
./launch.sh
EOF
chmod +x "$DESKTOP_LINK"
echo -e "${GREEN}✓ Created Desktop shortcut: ~/Desktop/EdukaAI-Studio.command${NC}"

# Installation complete
echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo " EdukaAI Studio is ready to use!"
echo ""
echo "📱 To start the application:"
echo "   • Double-click 'EdukaAI-Studio.command' on your Desktop"
echo "   • Or run: ${INSTALL_DIR}/launch.sh"
echo ""
echo " Getting Started:"
echo "   1. Open your browser to: http://localhost:3030"
echo "   2. Upload your training dataset"
echo "   3. Select a base model and training preset"
echo "   4. Click 'Start Training'"
echo ""
echo " Web page: https://eduka.elgap.ai"
echo " Documentation: ${INSTALL_DIR}/README.md"
echo " Support: https://github.com/elgap/edukaai-studio/issues"
echo ""
echo " Tip: Keep this terminal window open while using EdukaAI Studio"
echo ""
echo "Happy fine-tuning! "
echo ""

# Ask to launch now
if [ "$AUTO_YES" = false ]; then
    read_tty -p "Launch EdukaAI Studio now? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [ -z "$REPLY" ]]; then
        echo ""
        echo "🚀 Launching EdukaAI Studio..."
        echo "   Opening browser in a few seconds..."
        echo ""
        sleep 2
        ./launch.sh
    fi
else
    echo ""
    echo "🚀 Auto-launching EdukaAI Studio..."
    echo "   Opening browser in a few seconds..."
    echo ""
    sleep 2
    ./launch.sh
fi
