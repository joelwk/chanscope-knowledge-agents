#!/bin/bash

# Set essential environment variables
export PYTHONPATH=${PYTHONPATH}:${REPL_HOME}
export PYTHONUNBUFFERED=1

# Load environment variables
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
if [ -f "${ROOT_DIR}/.env" ]; then
    echo "Loading environment from .env file..."
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

# Get environment with fallback
ENV="${ENV:-development}"
echo "=== Starting in ${ENV} mode ==="

# Set environment-specific variables
if [ "$ENV" = "production" ]; then
    export REPLIT_ENV="production"
    export FASTAPI_ENV="production"
    export FASTAPI_DEBUG="false"
    export PORT="80"
    export UI_PORT="80"
    export API_PORT="80"
    export API_BASE_PATH="/api/v1"
else
    export REPLIT_ENV="development"
    export FASTAPI_ENV="development"
    export FASTAPI_DEBUG="true"
    export PORT="3001"
    export UI_PORT="3000"
    export API_PORT="3001"
    export API_BASE_PATH="/api/v1"
fi

export FASTAPI_APP="api.app:app"

# Enhanced service check with health endpoint
check_service() {
    local url=$1
    local max_attempts=$2
    local service_name=$3
    local attempt=1
    local retry_delay=5

    echo "Waiting for $service_name at $url..."
    while [ $attempt -le $max_attempts ]; do
        if curl -sSf "$url/health" >/dev/null 2>&1; then
            echo "$service_name is ready"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready..."
        sleep $retry_delay
        attempt=$((attempt + 1))
    done
    echo "$service_name failed to start"
    return 1
}

# Function to cleanup processes
cleanup() {
    echo "Cleaning up processes..."
    kill $API_PID 2>/dev/null
    kill $UI_PID 2>/dev/null
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start API service
echo "Starting API service..."
bash scripts/replit/api.sh &
API_PID=$!

# Wait for API to be ready
if ! check_service "http://localhost:${API_PORT}${API_BASE_PATH}" 12 "API"; then
    echo "API failed to start. Exiting..."
    cleanup
    exit 1
fi

# Start UI service
echo "Starting UI service..."
bash scripts/replit/ui.sh &
UI_PID=$!

# Wait for UI to be ready
if ! check_service "http://localhost:${UI_PORT}/ui" 12 "UI"; then
    echo "UI failed to start. Exiting..."
    cleanup
    exit 1
fi

# Monitor services
echo "All services started successfully. Monitoring..."
while true; do
    if ! kill -0 $UI_PID 2>/dev/null; then
        echo "UI service died unexpectedly"
        cleanup
        exit 1
    fi
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "API service died unexpectedly"
        cleanup
        exit 1
    fi
    sleep 5
done