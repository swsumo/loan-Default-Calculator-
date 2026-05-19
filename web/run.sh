#!/bin/bash
# Run the Fraud Detection web app
# Usage: bash web/run.sh

cd "$(dirname "$0")/.."

# Load .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

source venv/bin/activate

if [ -z "$GROQ_API_KEY" ]; then
  echo "Note: GROQ_API_KEY not set — AI explanation features disabled."
  echo "Add your key to .env file to enable them."
  echo ""
fi

echo "Starting Fraud Detection Ops at http://localhost:5001"
python web/app.py
