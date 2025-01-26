#!/bin/bash

# Start API server in background
./scripts/replit/api.sh &

# Start scheduled update process in background
./scripts/replit/scheduled_update.sh &

# Wait for API to be healthy
echo "Waiting for API to be ready..."
while ! curl -s http://0.0.0.0:5000/api/health_replit > /dev/null; do
    sleep 1
done
echo "API is ready"

# Start UI server
poetry run python -m chainlit run chainlit_frontend/app.py --host=0.0.0.0 --port=443