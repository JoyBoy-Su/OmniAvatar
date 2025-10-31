#!/bin/bash

# VAD Integration Setup Script
# This script installs the required VAD package for OmniAvatar

set -e  # Exit on error

echo "============================================"
echo "OmniAvatar VAD Integration Setup"
echo "============================================"
echo ""

# Check if we're in the correct directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found!"
    echo "Please run this script from the OmniAvatar root directory."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed!"
    echo "Please install Node.js and npm first."
    exit 1
fi

echo "✓ Found frontend directory"
echo "✓ npm is installed"
echo ""

# Navigate to frontend directory
cd frontend

echo "📦 Installing VAD package (@ricky0123/vad-react)..."
echo ""

# Install the package
npm install

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ VAD Setup Complete!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Start your backend server (if not already running)"
    echo "2. Start the frontend: cd frontend && npm run dev"
    echo "3. Open the app in Chrome 113+, Edge 113+, or Safari 18+"
    echo "4. Click the VAD toggle button and grant microphone permission"
    echo "5. Start speaking - VAD will auto-detect and record!"
    echo ""
    echo "📚 For more information, see:"
    echo "   - QUICK_START_VAD.md"
    echo "   - VAD_INTEGRATION_GUIDE.md"
    echo ""
else
    echo ""
    echo "❌ Installation failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi
