#!/usr/bin/env bash
set -e

echo "🚀 Setting up IntelliView AI Interview Platform..."

# Update package lists and install ffmpeg and other common dependencies for ML/CV
# libgl1-mesa-glx, libxext6, libsm6 are often required by OpenCV and other image/video libraries.
echo "Installing system dependencies (ffmpeg, libgl1-mesa-glx, libxext6, libsm6)..."
apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libxext6 libsm6

# Install Python dependencies into Render’s environment
# Use pip3 as specified in your original script
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Ensure uploads folder exists
mkdir -p uploads

echo "✅ Setup complete!"