#!/usr/bin/env bash
set -e
set -x

echo "🚀 Starting IntelliView AI Interview Platform…"
echo "Render PORT environment variable is: $PORT"

cd web_app

# Replace this shell with Gunicorn, binding immediately on the assigned port.
exec python3 -m gunicorn main:app \
  --bind 0.0.0.0:"${PORT:-10000}" \
  --workers 1 \
  --timeout 120 \
  --preload \
  --log-level info \
  --access-logfile - \
  --error-logfile -
