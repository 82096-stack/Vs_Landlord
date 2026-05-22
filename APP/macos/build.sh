#!/bin/bash
echo "========================================"
echo " DouDiZhu - macOS PyInstaller Build"
echo "========================================"
echo

# Check if pyinstaller is installed
if ! pip show pyinstaller &>/dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
fi

# Install project dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Clean previous builds
rm -rf build dist

# Build with spec file
echo "Building..."
pyinstaller main.spec

echo
echo "========================================"
echo "Build complete! Executable is in dist/DouDiZhu"
echo "========================================"
