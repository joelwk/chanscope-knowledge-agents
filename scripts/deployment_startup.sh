#!/bin/bash
# Deployment startup script for production
# This script starts the main API server and refresh dashboard for deployments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Starting Replit deployment startup from ${PROJECT_ROOT}..."

# Create essential directories
mkdir -p data temp_files data/stratified logs

# Create a temporary environment override file for deployment
cat > /tmp/deployment_env_override.sh << 'EOF'
export AUTO_CHECK_DATA="false"
export AUTO_PROCESS_DATA_ON_INIT="false"
export INIT_WAIT_TIME="20"
EOF

# Source the override file
source /tmp/deployment_env_override.sh

# Determine the serving port: prefer $PORT (Autoscale), fall back to $API_PORT, then 8080
TARGET_PORT="${PORT:-${API_PORT:-8080}}"
export API_PORT="${TARGET_PORT}"
export PORT="${TARGET_PORT}"

# Run the initialization in background after a short delay
INIT_DELAY="${INIT_DELAY:-5}"
echo "Scheduling background initialization after ${INIT_DELAY}s delay..."
(sleep "${INIT_DELAY}" && bash scripts/replit_init.sh 2>&1 | tee -a logs/init.log) &

# Start the main API server (health checks hit this port)
# Enable automatic refresh manager with configurable interval (default 3600 seconds = 1 hour)
echo "Starting main API server on port ${TARGET_PORT} with automatic database refresh..."
cd "${PROJECT_ROOT}" && \
  AUTO_CHECK_DATA=true \
  AUTO_REFRESH_MANAGER=true \
  DATA_REFRESH_INTERVAL=${DATA_REFRESH_INTERVAL:-3600} \
  uvicorn api.app:app --host 0.0.0.0 --port "${TARGET_PORT}" --log-level info
