#!/bin/bash

echo "🚀 Setting up IntelliView AI Interview Platform..."

# Navigate to project directory
cd "$(dirname "$0")"

# Check if Python 3.11 is installed and available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is not installed or not in PATH. Please install Python 3.11 first."
    exit 1
fi

echo "✅ Python 3.11 found"

# Create virtual environment (using python3.11 and a distinct name)
VENV_DIR=".venv_intelliview" # Define the new virtual environment directory name

echo "📦 Creating virtual environment in $VENV_DIR..."
python3.11 -m venv "$VENV_DIR"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR"/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip3 install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip3 install -r requirements.txt

# Create uploads directory for ATS feature
echo "📁 Creating uploads directory for ATS feature..."
mkdir -p uploads

echo "✅ Setup complete!"
echo ""
echo "To run the project in the future, just use: ./run.sh"
echo "Or manually: source $VENV_DIR/bin/activate && cd web_app && python main.py"
echo ""
echo "🌐 The application will be available at: http://localhost:8000"
echo "🔍 ATS Resume Scoring will be available at: http://localhost:8000/ats/"
