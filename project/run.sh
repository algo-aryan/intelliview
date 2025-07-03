#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed (FOR DEBUGGING)

echo "🚀 Starting IntelliView AI Interview Platform..."

# Print the PORT variable to see what Render is providing
echo "Render PORT environment variable is: $PORT"

# Change into the directory containing your Flask app
# This is crucial for 'main:app' to correctly resolve to web_app/main.py
echo "Changing directory to web_app..."
cd web_app

# Use Gunicorn to serve the Flask app
# The ${PORT:-10000} ensures a default port if Render’s PORT isn't set (e.g., during local testing)
echo "Attempting to start Gunicorn on 0.0.0.0:${PORT:-10000}..."

# Execute Gunicorn with increased timeout and verbose logging
# --workers 1: THIS IS THE KEY CHANGE TO REDUCE MEMORY USAGE AT STARTUP
# --timeout 120: Gives the app 120 seconds to start up (default is 30)
# --log-level info: Provides more detailed logs from Gunicorn itself
# --access-logfile -: Logs access to stdout
# --error-logfile -: Logs errors to stdout
exec gunicorn main:app \
    --bind 0.0.0.0:"${PORT:-10000}" \
    --workers 1 \
    --timeout 120 \
    --log-level info \
    --access-logfile - \
    --error-logfile -

echo "Gunicorn command finished. If you see this, Gunicorn exited prematurely."