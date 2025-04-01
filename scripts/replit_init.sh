#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Replit initialization script...${NC}"

# Ensure we're in the workspace root
WORKSPACE_ROOT="$PWD"
echo -e "${YELLOW}Workspace root: $WORKSPACE_ROOT${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$WORKSPACE_ROOT/data"
mkdir -p "$WORKSPACE_ROOT/data/stratified"
mkdir -p "$WORKSPACE_ROOT/logs"
mkdir -p "$WORKSPACE_ROOT/temp_files"

# Ensure Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}Poetry not found, installing...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Update Poetry dependencies
echo -e "${YELLOW}Installing/updating dependencies with Poetry...${NC}"
poetry install

# Check for required modules and add them if missing
for package in psycopg2-binary boto3 replit; do
    if ! poetry run python -c "import $package" &> /dev/null; then
        echo -e "${YELLOW}Adding $package to Poetry dependencies...${NC}"
        poetry add "$package"
    else
        echo -e "${GREEN}$package is already installed${NC}"
    fi
done

# Initialize PostgreSQL schema
echo -e "${YELLOW}Initializing PostgreSQL schema...${NC}"
if [ -n "$DATABASE_URL" ] || [ -n "$PGHOST" ]; then
    poetry run python -c "
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
poetry run python -c "
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
    poetry run python -c "
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

# Initialize data with stratification and embedding generation
echo -e "${YELLOW}Checking if data processing is needed...${NC}"
poetry run python scripts/process_data.py

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
    nohup poetry run python scripts/scheduled_update.py refresh --continuous --interval=$DATA_UPDATE_INTERVAL > "$WORKSPACE_ROOT/logs/scheduler.log" 2>&1 &
    
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

echo -e "${GREEN}Replit initialization completed successfully!${NC}"
echo -e "${YELLOW}You can now run the application with: poetry run python -m uvicorn api.app:app --host 0.0.0.0 --port 80${NC}"

# Verify root endpoint health check if app is already running
if curl -s http://localhost:80/ > /dev/null; then
    echo -e "${YELLOW}Testing root endpoint health check...${NC}"
    response=$(curl -s http://localhost:80/)
    if [[ $response == *"status"* ]]; then
        echo -e "${GREEN}Root endpoint health check is working correctly!${NC}"
        echo -e "${GREEN}Response: $response${NC}"
    else
        echo -e "${RED}Root endpoint health check is not returning the expected response${NC}"
        echo -e "${RED}Response: $response${NC}"
    fi
else
    echo -e "${YELLOW}Application is not running, skipping root endpoint health check${NC}"
fi 