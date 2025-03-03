#!/bin/bash

# Get the application root directory
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${APP_ROOT}/data"
LOGS_DIR="${DATA_DIR}/logs"
SCHEDULER_LOG="${LOGS_DIR}/scheduler.log"

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Get update interval from environment or use default (1 hour)
UPDATE_INTERVAL="${DATA_UPDATE_INTERVAL:-3600}"

echo "Starting Replit data scheduler with update interval: ${UPDATE_INTERVAL} seconds"
echo "Following Chanscope approach for data processing"

# Log environment information
echo "[$(date)] Starting Chanscope data scheduler in Replit environment" >> "${SCHEDULER_LOG}"
echo "[$(date)] Update interval: ${UPDATE_INTERVAL} seconds" >> "${SCHEDULER_LOG}"
echo "[$(date)] Python version: $(python --version 2>&1)" >> "${SCHEDULER_LOG}"
echo "[$(date)] Replit ID: ${REPL_ID}" >> "${SCHEDULER_LOG}"

# Create a marker file to indicate the scheduler is running
echo "Started: $(date)" > "${DATA_DIR}/.scheduler_running"
echo "Environment: replit" >> "${DATA_DIR}/.scheduler_running"
echo "Replit ID: ${REPL_ID}" >> "${DATA_DIR}/.scheduler_running"
echo "PID: $$" >> "${DATA_DIR}/.scheduler_running"

# Function to handle errors
handle_error() {
    local exit_code=$1
    local error_message=$2
    echo "[$(date)] ERROR: ${error_message} (exit code: ${exit_code})" >> "${SCHEDULER_LOG}"
    # Wait a bit before retrying to avoid rapid failure loops
    sleep 60
}

# Run the update script using Poetry in a loop
while true; do
    echo "[$(date)] Running scheduled data update in Replit environment..." >> "${SCHEDULER_LOG}" 2>&1
    
    # First run - check if this is initial startup
    if [ ! -f "${DATA_DIR}/complete_data.csv" ]; then
        echo "[$(date)] Initial startup detected. Following Chanscope two-phase approach..." >> "${SCHEDULER_LOG}" 2>&1
        
        # Phase 1: Load and stratify data, but skip embeddings (faster startup)
        echo "[$(date)] Phase 1: Loading and stratifying data..." >> "${SCHEDULER_LOG}" 2>&1
        cd "${APP_ROOT}" && poetry run python scripts/scheduled_update.py --run_once >> "${SCHEDULER_LOG}" 2>&1
        if [ $? -ne 0 ]; then
            handle_error $? "Phase 1 failed"
            continue
        fi
        
        # Phase 2: Generate embeddings separately
        echo "[$(date)] Phase 2: Generating embeddings..." >> "${SCHEDULER_LOG}" 2>&1
        cd "${APP_ROOT}" && poetry run python -c "
import asyncio
from knowledge_agents.data_ops import DataConfig, DataOperations
from config.settings import Config
from pathlib import Path

async def generate_embeddings():
    # Get settings from Config
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create data config
    data_config = DataConfig(
        root_data_path=Path(paths['root_data_path']),
        stratified_data_path=Path(paths['stratified']),
        temp_path=Path(paths['temp']),
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings['time_column'],
        strata_column=column_settings['strata_column']
    )
    
    # Initialize data operations
    operations = DataOperations(data_config)
    
    # Generate embeddings
    result = await operations.generate_embeddings()
    print(f'Embedding generation completed: {result}')

asyncio.run(generate_embeddings())
" >> "${SCHEDULER_LOG}" 2>&1
        if [ $? -ne 0 ]; then
            handle_error $? "Phase 2 failed"
            continue
        fi
    else
        # Regular update - follow Chanscope incremental update approach
        echo "[$(date)] Running incremental update (force_refresh=False)..." >> "${SCHEDULER_LOG}" 2>&1
        cd "${APP_ROOT}" && poetry run python scripts/scheduled_update.py --run_once >> "${SCHEDULER_LOG}" 2>&1
        if [ $? -ne 0 ]; then
            handle_error $? "Incremental update failed"
            continue
        fi
    fi
    
    # Log completion and next run time
    now=$(date)
    next_run=$(date -d "+${UPDATE_INTERVAL} seconds")
    echo "[${now}] Scheduled update completed. Next run at approximately: ${next_run}" >> "${SCHEDULER_LOG}" 2>&1
    
    # Update the marker file with last successful run
    echo "Last successful run: $(date)" >> "${DATA_DIR}/.scheduler_running"
    
    # Wait for the configured interval before next update
    sleep ${UPDATE_INTERVAL}
done