#!/bin/bash
# Deployment startup script for production
# This script starts the main API server on port 80 for health checks
# and runs the dashboard as a background process

echo "Starting Replit deployment startup..."

# Create essential directories
mkdir -p data temp_files data/stratified logs

# Run the initialization in background after a delay
(sleep 30 && python scripts/replit_init.py 2>&1 | tee -a logs/init.log) &

# Start the dashboard server on port 5001 in background (not exposed externally)
echo "Starting dashboard server on port 5001 (background)..."
PORT=5001 python api/refresh_dashboard.py 2>&1 | tee -a logs/dashboard.log &

# Start the main API server on port 80 (this is what health checks will hit)
echo "Starting main API server on port 80 (foreground)..."
python api/app.py