#!/usr/bin/env bash
# One-shot launcher. Equivalent of `npm start` for this project.
# Usage: ./start.sh
set -e

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in your Supabase values:"
  echo "  cp .env.example .env"
  exit 1
fi

if [ ! -x venv/bin/python3.14 ]; then
  echo "ERROR: venv/bin/python3.14 not found. The venv is missing or broken."
  exit 1
fi

exec ./venv/bin/python3.14 app.py
