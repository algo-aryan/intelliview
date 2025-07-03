#!/usr/bin/env bash
set -e

echo "Installing FFmpeg..."
apt-get update && apt-get install -y ffmpeg

echo "🚀 Setting up IntelliView AI Interview Platform..."

# Install Python dependencies into Render’s environment (or your local venv)
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Ensure uploads folder exists
mkdir -p uploads

echo "✅ Setup complete!"
