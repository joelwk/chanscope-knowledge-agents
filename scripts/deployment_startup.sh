#!/bin/bash
# Deployment startup script for production
# This script starts the main API server on port 5000 for health checks

echo "Starting Replit deployment startup..."

# Create essential directories
mkdir -p data temp_files data/stratified logs

# Create a temporary environment override file for deployment
cat > /tmp/deployment_env_override.sh << 'EOF'
export AUTO_CHECK_DATA="false"
export AUTO_PROCESS_DATA_ON_INIT="false"
export API_PORT="5000"
export INIT_WAIT_TIME="20"
EOF

# Source the override file
source /tmp/deployment_env_override.sh

# Run the initialization in background after a longer delay
(sleep 60 && bash scripts/replit_init.sh 2>&1 | tee -a logs/init.log) &

# Start the main API server on port 5000 (this is what health checks will hit)
# Enable automatic refresh manager with configurable interval (default 3600 seconds = 1 hour)
echo "Starting main API server on port 5000 with automatic database refresh..."
cd /home/runner/workspace && \
  AUTO_CHECK_DATA=true \
  AUTO_REFRESH_MANAGER=true \
  DATA_REFRESH_INTERVAL=${DATA_REFRESH_INTERVAL:-3600} \
  API_PORT=5000 \
  python -m api.app