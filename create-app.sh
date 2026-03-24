#!/bin/bash

# Create a macOS Application Bundle for EdukaAI Studio
# This creates a .app that can be double-clicked to launch

set -e

echo "🍎 Creating macOS Application Bundle"
echo "====================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

APP_NAME="EdukaAI Studio"
APP_BUNDLE="EdukaAI Studio.app"
VERSION=${1:-"1.0.0"}

echo "📦 App Name: $APP_NAME"
echo "📦 Version: $VERSION"
echo ""

# Remove old bundle if exists
if [ -d "$APP_BUNDLE" ]; then
    echo "Removing old app bundle..."
    rm -rf "$APP_BUNDLE"
fi

# Create app bundle structure
echo "📁 Creating app bundle structure..."
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>EdukaAI Studio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>ai.edukai.studio</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Create launcher script
cat > "$APP_BUNDLE/Contents/MacOS/EdukaAI Studio" << 'EOF'
#!/bin/bash

# Get the directory where the app is located
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
cd "$APP_DIR"

# Check if first run
if [ ! -d "backend/.venv" ]; then
    # Show installation dialog
    osascript -e 'display dialog "EdukaAI Studio needs to be installed. This may take a few minutes." buttons {"Cancel", "Install"} default button "Install"' 2>/dev/null
    
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # Run installer
    ./install.sh
    
    if [ $? -ne 0 ]; then
        osascript -e 'display dialog "Installation failed. Please check the terminal output." buttons {"OK"} default button "OK"' 2>/dev/null
        exit 1
    fi
    
    osascript -e 'display dialog "Installation complete! EdukaAI Studio will now launch." buttons {"OK"} default button "OK"' 2>/dev/null
fi

# Open Terminal and run the application
osascript << 'APPLESCRIPT'
tell application "Terminal"
    activate
    set window1 to do script "cd '" & POSIX path of (path to me) & "'; cd ../../../; ./launch.sh"
    set custom title of window1 to "EdukaAI Studio"
end tell
APPLESCRIPT

exit 0
EOF

chmod +x "$APP_BUNDLE/Contents/MacOS/EdukaAI Studio"

# Try to copy icon if it exists
if [ -f "docs/icon.icns" ]; then
    cp "docs/icon.icns" "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
    echo "✓ App icon added"
else
    echo "⚠️  No app icon found (create docs/icon.icns to add one)"
fi

# Set app metadata
xattr -cr "$APP_BUNDLE" 2>/dev/null || true

echo ""
echo "✅ Application Bundle Created!"
echo "================================"
echo ""
echo "📱 Location: $APP_BUNDLE"
echo ""
echo "To install:"
echo "1. Drag $APP_BUNDLE to your Applications folder"
echo "2. Double-click to launch"
echo "3. First run will install dependencies (may take a few minutes)"
echo ""
echo "To distribute:"
echo "  - Right-click on $APP_BUNDLE"
echo "  - Select 'Compress' to create .zip"
echo "  - Or run: tar -czf 'EdukaAI Studio.app.tar.gz' '$APP_BUNDLE'"
echo ""
