#!/bin/bash

echo "🚀 Starting IntelliView AI Interview Platform..."

# Navigate to project directory
cd "$(dirname "$0")"

# Define the virtual environment directory name (must match setup.sh)
VENV_DIR=".venv_intelliview"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment '$VENV_DIR' not found. Running setup first..."
    ./setup.sh
    # Exit if setup failed
    if [ ! -d "$VENV_DIR" ]; then
        echo "❌ Setup failed to create virtual environment. Exiting."
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR"/bin/activate

# Navigate to app directory
# Assuming main.py is in web_app/ relative to run.sh
if [ -d "web_app" ]; then
    cd web_app
else
    echo "Error: 'web_app' directory not found. Please run this script from the project root."
    exit 1
fi

# Check if all dependencies are installed (simplified check)
echo "🔍 Checking dependencies..."
# Check for a critical module from requirements.txt, e.g., Flask
python -c "import flask" 2>/dev/null || {
    echo "❌ Some dependencies are missing. Please run setup again or install manually."
    exit 1
}

echo "✅ All dependencies are installed"

# Create uploads directory if it doesn't exist
if [ ! -d "../uploads" ]; then
    echo "📁 Creating uploads directory..."
    mkdir -p ../uploads
fi

echo "🌟 Starting the application..."
echo "🌐 Open your browser and go to: http://localhost:8000"
echo "🔍 ATS Resume Scoring available at: http://localhost:8000/ats/"
echo "📱 Press Ctrl+C to stop the server"
echo ""

# Run the Flask application
python3 main.py
