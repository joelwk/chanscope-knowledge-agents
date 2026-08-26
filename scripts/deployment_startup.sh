#!/bin/bash
# Deployment startup script for production
# This script starts the main API server and mounted frontends for deployments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Starting Replit deployment startup from ${PROJECT_ROOT}..."

# Fix for libstdc++.so.6 - set library path for C++ dependencies
# Use nix-shell to find the gcc lib path efficiently
GCC_LIB_PATH=$(nix-instantiate --eval -E 'with import <nixpkgs> {}; "${stdenv.cc.cc.lib}/lib"' 2>/dev/null | tr -d '"')
if [ -z "$GCC_LIB_PATH" ] || [ ! -d "$GCC_LIB_PATH" ]; then
    # Fallback: search for libstdc++ with timeout
    GCC_LIB_PATH=$(timeout 5 find /nix/store -maxdepth 4 -name "libstdc++.so.6" -type f 2>/dev/null | grep gcc | head -1 | xargs dirname 2>/dev/null)
fi
if [ -n "$GCC_LIB_PATH" ] && [ -d "$GCC_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="${GCC_LIB_PATH}:${LD_LIBRARY_PATH}"
    echo "Set LD_LIBRARY_PATH to include: ${GCC_LIB_PATH}"
else
    echo "Warning: Could not locate libstdc++ library path"
fi

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
echo "Frontend URLs:"
echo "  - Refresh dashboard: /refresh"
echo "  - Query console:     /query-ui"
cd "${PROJECT_ROOT}" && \
  AUTO_CHECK_DATA=true \
  AUTO_REFRESH_MANAGER=true \
  DATA_REFRESH_INTERVAL=${DATA_REFRESH_INTERVAL:-3600} \
  uvicorn api.app:app --host 0.0.0.0 --port "${TARGET_PORT}" --log-level info
