#!/bin/bash

# Environment should be passed from start.sh
ENV="${ENV:-development}"

# Load common environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

if [ "$ENV" = "production" ]; then
    echo "Starting API in PRODUCTION mode"
    export PORT="80"
    export API_PORT="80"
    export REPLIT_ENV="production"
    export FASTAPI_ENV="production"
    export FASTAPI_DEBUG="false"
    export API_BASE_PATH="/api/v1"

    echo "API Configuration:"
    echo "- Port: ${API_PORT}"
    echo "- Environment: production"
    echo "- Base Path: ${API_BASE_PATH}"

    # Start the API server in production mode
    exec poetry run python -m uvicorn api.app:app \
        --host "0.0.0.0" \
        --port "${API_PORT}" \
        --log-level info \
        --workers 4 \
        --timeout-keep-alive 20
else
    echo "Starting API in DEVELOPMENT mode"
    export PORT="3001"
    export API_PORT="3001"
    export REPLIT_ENV="development"
    export FASTAPI_ENV="development"
    export FASTAPI_DEBUG="true"
    export API_BASE_PATH="/api/v1"

    echo "API Configuration:"
    echo "- Port: ${API_PORT}"
    echo "- Environment: development"
    echo "- Base Path: ${API_BASE_PATH}"

    # Start the API server in development mode with hot reload
    exec poetry run python -m uvicorn api.app:app \
        --host "0.0.0.0" \
        --port "${API_PORT}" \
        --log-level debug \
        --reload \
        --reload-dir api \
        --reload-dir config
fi
