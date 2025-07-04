#!/usr/bin/env bash
set -e

echo "🚀 Setting up IntelliView AI Interview Platform..."

# Update and install only available system packages for current distro
echo "Installing system dependencies (ffmpeg, libgl1, libxext6, libsm6)..."
apt-get update && apt-get install -y ffmpeg libgl1 libxext6 libsm6

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Ensure uploads folder exists
mkdir -p uploads

echo "✅ Setup complete!"