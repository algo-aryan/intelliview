#!/usr/bin/env bash
set -e

echo "🚀 Starting IntelliView AI Interview Platform..."

# Change into the directory containing your Flask app
cd web_app

# Use Gunicorn to serve the Flask app, binding to Render’s provided PORT (or 10000 locally)
# exec gunicorn main:app --bind 0.0.0.0:"${PORT:-10000}" --workers 4
python3 main.py