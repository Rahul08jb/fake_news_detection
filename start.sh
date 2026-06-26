#!/usr/bin/env sh
set -e

export PORT=${PORT:-5000}
export PYTHONPATH=./

gunicorn fake_news_detection.app:app \
  --bind 0.0.0.0:${PORT} \
  --workers 2 \
  --timeout 120

