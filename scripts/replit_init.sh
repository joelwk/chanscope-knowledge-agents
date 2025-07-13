#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting lightweight Replit initialization script...${NC}"

# Ensure local packages take precedence over system packages
export PYTHONPATH="$PWD/.pythonlibs/lib/python3.11/site-packages:${PYTHONPATH:-}"

# ==============================================================================
# CRITICAL SECTION: Fast startup essentials only
# ==============================================================================

# Create essential directories immediately
echo -e "${YELLOW}Creating essential directories...${NC}"
mkdir -p "$PWD/logs"
mkdir -p "$PWD/temp_files"
mkdir -p "$PWD/data"
mkdir -p "$PWD/data/stratified"

# Ensure scripts/utils directory exists
mkdir -p "$PWD/scripts/utils"
touch "$PWD/scripts/utils/__init__.py"

echo -e "${GREEN}Essential directories created successfully.${NC}"

# Quick dependency check - don't install, just verify critical ones exist
echo -e "${YELLOW}Checking critical dependencies...${NC}"
python3 -c "
try:
    import fastapi, uvicorn, pandas, psycopg2
    print('Critical dependencies available')
except ImportError as e:
    print(f'Warning: Missing dependency {e}')
    print('Will attempt installation in background...')
" || echo -e "${YELLOW}Some dependencies may need installation${NC}"

# ==============================================================================
# BACKGROUND INITIALIZATION: Run heavy tasks in background
# ==============================================================================
echo -e "${YELLOW}Starting background initialization tasks...${NC}"

# Create a background initialization function
(
    # Wait a bit for the server to fully stabilize
    echo -e "${YELLOW}Waiting 10 seconds for server stabilization...${NC}"
    sleep 10

    # Now do the heavy lifting
    echo -e "${YELLOW}Installing/updating project dependencies...${NC}"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --quiet --no-warn-script-location
        echo -e "${GREEN}Dependencies installed successfully.${NC}"
    else
        echo -e "${RED}Warning: requirements.txt not found!${NC}"
    fi

    # Initialize PostgreSQL schema
    echo -e "${YELLOW}Initializing PostgreSQL schema...${NC}"
    if [ -n "$DATABASE_URL" ] || [ -n "$PGHOST" ]; then
        python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from config.replit import PostgresDB
    db = PostgresDB()
    db.initialize_schema()
    print('PostgreSQL schema initialized successfully')
except Exception as e:
    print(f'Error initializing PostgreSQL schema: {e}')
"
    else
        echo -e "${RED}No PostgreSQL database connection information found${NC}"
    fi

    # Verify services
    echo -e "${YELLOW}Verifying services...${NC}"
    
    # Verify Replit Key-Value store
    python3 -c "
try:
    from replit import db
    db['test_key'] = 'test_value'
    assert db['test_key'] == 'test_value'
    del db['test_key']
    print('Replit Key-Value store is functioning correctly')
except Exception as e:
    print(f'Error with Replit Key-Value store: {e}')
" || echo -e "${YELLOW}Replit KV store may not be available${NC}"

    # Verify AWS credentials if present
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
    bucket = os.environ.get('S3_BUCKET')
    prefix = os.environ.get('S3_BUCKET_PREFIX', 'data/')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    if 'Contents' in response:
        print(f'AWS credentials verified successfully, found objects in {bucket}/{prefix}')
    else:
        print(f'AWS credentials verified but no objects found in {bucket}/{prefix}')
except Exception as e:
    print(f'Error verifying AWS credentials: {e}')
" || echo -e "${YELLOW}AWS services may not be available${NC}"
    fi

    # Data processing - only if explicitly requested
    echo -e "${YELLOW}Checking if data processing is needed...${NC}"
    
    # Use a more lightweight check for data processing needs
    SHOULD_PROCESS_DATA=$(python3 -c "
import sys
import os
sys.path.insert(0, '.')

# Check if we should run data processing
auto_process = os.environ.get('AUTO_PROCESS_DATA_ON_INIT', 'false').lower() in ('true', '1', 'yes')
force_refresh = os.environ.get('FORCE_DATA_REFRESH', 'false').lower() in ('true', '1', 'yes')

if force_refresh:
    print('YES')
elif auto_process:
    try:
        from scripts.utils.processing_lock import ProcessLockManager
        lock_manager = ProcessLockManager()
        needs_init, marker_data = lock_manager.check_initialization_status()
        if needs_init:
            print('YES')
        else:
            print('NO')
    except Exception as e:
        print('NO')  # Default to no processing if check fails
else:
    print('NO')
")

    if [ "$SHOULD_PROCESS_DATA" = "YES" ]; then
        echo -e "${YELLOW}Starting data processing in background...${NC}"
        nohup python3 scripts/process_data.py > "$PWD/logs/data_processing.log" 2>&1 &
        echo -e "${GREEN}Data processing started in background${NC}"
    else
        echo -e "${GREEN}Skipping data processing (use AUTO_PROCESS_DATA_ON_INIT=true to enable)${NC}"
    fi

    # Configure scheduler if enabled
    ENABLE_DATA_SCHEDULER="${ENABLE_DATA_SCHEDULER:-false}"
    if [ "$ENABLE_DATA_SCHEDULER" = "true" ]; then
        DATA_UPDATE_INTERVAL="${DATA_UPDATE_INTERVAL:-3600}"
        echo -e "${YELLOW}Starting data scheduler with interval: ${DATA_UPDATE_INTERVAL}s${NC}"
        
        # Clean up any existing scheduler
        SCHEDULER_PID_FILE="$PWD/data/.scheduler_pid"
        if [ -f "$SCHEDULER_PID_FILE" ]; then
            rm -f "$SCHEDULER_PID_FILE"
        fi
        
        # Start scheduler
        nohup python3 scripts/scheduled_update.py refresh --continuous --interval=$DATA_UPDATE_INTERVAL > "$PWD/logs/scheduler.log" 2>&1 &
        SCHEDULER_PID=$!
        echo $SCHEDULER_PID > "$SCHEDULER_PID_FILE"
        echo -e "${GREEN}Data scheduler started with PID: $SCHEDULER_PID${NC}"
    fi

    # Mark completion
    echo "Background initialization completed at $(date)" > "$PWD/data/.replit_init_complete"
    echo -e "${GREEN}Background initialization completed successfully!${NC}"
    
) > "$PWD/logs/replit_init_background.log" 2>&1 &

BACKGROUND_PID=$!
echo -e "${GREEN}Background initialization started with PID: $BACKGROUND_PID${NC}"
echo -e "${GREEN}Logs available at: $PWD/logs/replit_init_background.log${NC}"

# Quick exit - server should already be running by now
echo -e "${GREEN}Lightweight initialization completed. Server is ready!${NC}"
exit 0 