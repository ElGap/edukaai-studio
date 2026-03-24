#!/bin/bash

# Build script for creating distributable packages
# This creates a clean package without dev dependencies and cache files

set -e

echo "🔨 Building EdukaAI Studio Distribution Package"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Version
VERSION=${1:-"1.0.0"}
echo "📦 Version: $VERSION"
echo ""

# Create build directory
BUILD_DIR="build/edukai-studio-$VERSION"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "📁 Creating build structure..."

# Copy essential files
cp -r backend "$BUILD_DIR/"
cp -r frontend "$BUILD_DIR/"
cp -r docs "$BUILD_DIR/" 2>/dev/null || true
cp README.md "$BUILD_DIR/"
cp LICENSE "$BUILD_DIR/"
cp CONTRIBUTING.md "$BUILD_DIR/"
cp install.sh "$BUILD_DIR/"
cp launch.sh "$BUILD_DIR/"

# Clean up backend
echo "🧹 Cleaning backend..."
cd "$BUILD_DIR/backend"
rm -rf .venv
rm -rf __pycache__
rm -rf **/__pycache__
rm -rf **/*.pyc
rm -rf **/*.pyo
rm -rf storage/datasets/*
rm -rf storage/runs/*
rm -rf storage/db/*.db
rm -rf storage/db/*.db-journal
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov
rm -rf .tox
rm -rf dist
rm -rf build
rm -rf *.egg-info

# Keep directory structure
mkdir -p storage/datasets
mkdir -p storage/runs/downloaded_models
mkdir -p storage/db
mkdir -p storage/exports
touch storage/datasets/.gitkeep
touch storage/runs/.gitkeep

# Clean up frontend
echo "🧹 Cleaning frontend..."
cd "$BUILD_DIR/frontend"
rm -rf node_modules
rm -rf dist
rm -rf dist-ssr
rm -rf .env.local

# Create archive
echo "📦 Creating archive..."
cd "$SCRIPT_DIR/build"
tar -czf "edukai-studio-$VERSION.tar.gz" "edukai-studio-$VERSION"

# Calculate size
SIZE=$(du -h "edukai-studio-$VERSION.tar.gz" | cut -f1)

echo ""
echo "✅ Build Complete!"
echo "=================="
echo ""
echo "📦 Package: build/edukai-studio-$VERSION.tar.gz"
echo "📊 Size: $SIZE"
echo ""
echo "To distribute:"
echo "1. Upload to GitHub Releases"
echo "2. Share the .tar.gz file"
echo ""
echo "Users can install with:"
echo "  tar -xzf edukai-studio-$VERSION.tar.gz"
echo "  cd edukai-studio-$VERSION"
echo "  ./install.sh"
echo "  ./launch.sh"
echo ""
