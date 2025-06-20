#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Replit initialization script...${NC}"

# Ensure local packages take precedence over system packages
export PYTHONPATH="$PWD/.pythonlibs/lib/python3.11/site-packages:${PYTHONPATH:-}"

# ==============================================================================
# FOREGROUND INITIALIZATION: Ensure environment and dependencies are ready.
# ==============================================================================

# Install dependencies synchronously before starting background tasks
echo -e "${YELLOW}Installing/updating project dependencies with pip...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi


echo -e "${YELLOW}Starting background initialization tasks...${NC}"
# ==============================================================================
# BACKGROUND INITIALIZATION: Run non-blocking tasks.
# ==============================================================================
(
    # Ensure we're in the workspace root
    WORKSPACE_ROOT="$PWD"
    echo -e "${YELLOW}Workspace root: $WORKSPACE_ROOT${NC}"

    # Create necessary directories
    echo -e "${YELLOW}Creating directories...${NC}"
    mkdir -p "$WORKSPACE_ROOT/data"
    mkdir -p "$WORKSPACE_ROOT/data/stratified"
    mkdir -p "$WORKSPACE_ROOT/logs"
    mkdir -p "$WORKSPACE_ROOT/temp_files"

    # Ensure scripts/utils directory exists
    mkdir -p "$WORKSPACE_ROOT/scripts/utils"
    touch "$WORKSPACE_ROOT/scripts/utils/__init__.py"

    # Initialize PostgreSQL schema
    echo -e "${YELLOW}Initializing PostgreSQL schema...${NC}"
    if [ -n "$DATABASE_URL" ] || [ -n "$PGHOST" ]; then
        python3 -c "
import asyncio
from config.replit import PostgresDB

def initialize_db():
    try:
        db = PostgresDB()
        db.initialize_schema()
        print('PostgreSQL schema initialized successfully')
    except Exception as e:
        print(f'Error initializing PostgreSQL schema: {e}')

# Run the function
initialize_db()
"
    else
        echo -e "${RED}No PostgreSQL database connection information found${NC}"
        echo -e "${YELLOW}Please set up a PostgreSQL database in Replit and try again${NC}"
    fi

    # Verify Replit Key-Value store
    echo -e "${YELLOW}Verifying Replit Key-Value store access...${NC}"
    python3 -c "
try:
    from replit import db
    db['test_key'] = 'test_value'
    assert db['test_key'] == 'test_value'
    del db['test_key']
    print('Replit Key-Value store is functioning correctly')
except Exception as e:
    print(f'Error with Replit Key-Value store: {e}')
"

    # Verify AWS credentials
    echo -e "${YELLOW}Verifying AWS credentials...${NC}"
    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ] && [ -n "$S3_BUCKET" ]; then
        python3 -c "
import boto3
import os

try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    )

    # Try listing objects to verify credentials
    bucket = os.environ.get('S3_BUCKET')
    prefix = os.environ.get('S3_BUCKET_PREFIX', 'data/')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)

    if 'Contents' in response:
        print(f'AWS credentials verified successfully, found objects in {bucket}/{prefix}')
    else:
        print(f'AWS credentials verified but no objects found in {bucket}/{prefix}')

except Exception as e:
    print(f'Error verifying AWS credentials: {e}')
"
    else
        echo -e "${RED}AWS credentials not fully configured${NC}"
        echo -e "${YELLOW}Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET in Secrets${NC}"
    fi

    # Give the server more time to finish its health checks before starting heavy data processing
    echo -e "${YELLOW}Waiting for server to stabilize before starting data processing...${NC}"
    sleep 30

    # Check if initialization was recently completed using the ProcessLockManager
    echo -e "${YELLOW}Checking if data processing is needed...${NC}"
    PROCESS_CHECK=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from scripts.utils.processing_lock import ProcessLockManager

    # Create lock manager and check status
    lock_manager = ProcessLockManager()
    needs_init, marker_data = lock_manager.check_initialization_status()

    if not needs_init and marker_data:
        completion_time = marker_data.get('completion_time', 'unknown time')
        print(f'SKIP: Previous initialization completed successfully at {completion_time}')
    else:
        if marker_data and marker_data.get('status') == 'error':
            error = marker_data.get('error', 'unknown error')
            print(f'RUN: Previous initialization failed with error: {error}')
        else:
            print('RUN: Initialization needed')

except ImportError as e:
    print(f'RUN: Could not import ProcessLockManager: {e}')
except Exception as e:
    print(f'RUN: Error checking initialization status: {e}')
    import traceback
    traceback.print_exc()
")

    # Check if we should skip or run data processing
    if [[ $PROCESS_CHECK == SKIP* ]]; then
        echo -e "${GREEN}${PROCESS_CHECK#SKIP: }${NC}"
        echo -e "${GREEN}Skipping data processing${NC}"
    else
        echo -e "${YELLOW}${PROCESS_CHECK#RUN: }${NC}"
        echo -e "${YELLOW}Starting data processing in background...${NC}"

        # Use nohup to keep the process running even if the parent is terminated
        nohup python3 scripts/process_data.py > "$WORKSPACE_ROOT/logs/data_processing.log" 2>&1 &
        DATA_PROCESS_PID=$!
        echo -e "${GREEN}Data processing started in background with PID: $DATA_PROCESS_PID${NC}"
        echo -e "${GREEN}Processing logs available at: $WORKSPACE_ROOT/logs/data_processing.log${NC}"
    fi

    # Configure and start the data scheduler if enabled
    ENABLE_DATA_SCHEDULER="${ENABLE_DATA_SCHEDULER:-true}"
    DATA_UPDATE_INTERVAL="${DATA_UPDATE_INTERVAL:-3600}"

    if [ "$ENABLE_DATA_SCHEDULER" = "true" ]; then
        echo -e "${YELLOW}Setting up the data scheduler...${NC}"

        # Check if scheduler is already running
        SCHEDULER_PID_FILE="$WORKSPACE_ROOT/data/.scheduler_pid"

        if [ -f "$SCHEDULER_PID_FILE" ]; then
            echo -e "${YELLOW}Cleaning up previous scheduler instance...${NC}"
            rm -f "$SCHEDULER_PID_FILE"
        fi

        # Start the scheduler in background using nohup
        echo -e "${YELLOW}Starting data scheduler with interval: ${DATA_UPDATE_INTERVAL}s${NC}"

        # Create a background task that runs the scheduler
        nohup python3 scripts/scheduled_update.py refresh --continuous --interval=$DATA_UPDATE_INTERVAL > "$WORKSPACE_ROOT/logs/scheduler.log" 2>&1 &

        # Save the PID
        SCHEDULER_PID=$!
        echo $SCHEDULER_PID > "$SCHEDULER_PID_FILE"

        echo -e "${GREEN}Data scheduler started with PID: $SCHEDULER_PID${NC}"
        echo -e "${GREEN}Scheduler will update data every ${DATA_UPDATE_INTERVAL} seconds${NC}"
        echo -e "${YELLOW}Scheduler logs available at: $WORKSPACE_ROOT/logs/scheduler.log${NC}"
    else
        echo -e "${YELLOW}Data scheduler is disabled. Set ENABLE_DATA_SCHEDULER=true to enable automatic updates.${NC}"
    fi

    # Create initialization marker
    echo -e "${YELLOW}Creating initialization markers...${NC}"
    echo "Initialized at $(date)" > "$WORKSPACE_ROOT/data/.replit_init_complete"

    echo -e "${GREEN}Replit background initialization completed successfully!${NC}"
) > ./logs/replit_init.log 2>&1 &

# Report that the background initialization has started
echo "Background initialization process started. API server should already be running."
echo "Initialization logs available at: ./logs/replit_init.log"

# Exit cleanly with success code - this enables the server which should already be running
exit 0 