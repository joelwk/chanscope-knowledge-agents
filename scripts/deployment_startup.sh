#!/bin/bash
# Deployment startup script for production
# This script starts the main API server on port 5000 for health checks

echo "Starting Replit deployment startup..."

# Create essential directories
mkdir -p data temp_files data/stratified logs

# Export environment variables for quick startup
export AUTO_CHECK_DATA=false
export AUTO_PROCESS_DATA_ON_INIT=false
export API_PORT=5000

# Run the initialization in background after a delay
(sleep 45 && python scripts/replit_init.py 2>&1 | tee -a logs/init.log) &

# Start the main API server on port 5000 (this is what health checks will hit)
echo "Starting main API server on port 5000..."
cd /home/runner/workspace && python -m api.app