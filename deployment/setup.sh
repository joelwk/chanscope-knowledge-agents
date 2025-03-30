#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Read custom environment variables related to startup behavior
RUN_TESTS_ON_STARTUP=${RUN_TESTS_ON_STARTUP:-false}
TEST_TYPE=${TEST_TYPE:-all}
AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-true}
ABORT_ON_TEST_FAILURE=${ABORT_ON_TEST_FAILURE:-false}
TEST_RESULTS_DIR=${TEST_RESULTS_DIR:-"$PWD/test_results"}

# Data scheduler configuration
ENABLE_DATA_SCHEDULER=${ENABLE_DATA_SCHEDULER:-true}
DATA_UPDATE_INTERVAL=${DATA_UPDATE_INTERVAL:-3600}  # 1 hour in seconds

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine if we're in a Replit environment
if [ -n "$REPL_ID" ] || [ "$REPLIT_ENV" = "replit" ] || [ "$REPLIT_ENV" = "true" ] || [ "$REPLIT_ENV" = "production" ]; then
    IS_REPLIT=true
    echo -e "${YELLOW}Detected Replit environment (REPL_ID: $REPL_ID, REPL_SLUG: $REPL_SLUG)${NC}"
else
    IS_REPLIT=false
    echo -e "${YELLOW}Not in Replit environment, assuming Docker or local${NC}"
fi

# Determine application root directory based on environment
if [ "$IS_REPLIT" = true ]; then
    # In Replit, use the workspace directory
    APP_ROOT="$PWD"
    DATA_DIR="$APP_ROOT/data"
    LOGS_DIR="$APP_ROOT/logs"
    TEMP_DIR="$APP_ROOT/temp_files"
    POETRY_CACHE_DIR="$APP_ROOT/.cache/pypoetry"
    echo -e "${YELLOW}Running in Replit environment. App root: $APP_ROOT${NC}"
else
    # In Docker or local, use /app if it exists and is writable, otherwise use current directory
    if [ -d "/app" ] && [ -w "/app" ]; then
        APP_ROOT="/app"
        DATA_DIR="/app/data"
        LOGS_DIR="/app/logs"
        TEMP_DIR="/app/temp_files"
        POETRY_CACHE_DIR="/home/nobody/.cache/pypoetry"
        echo -e "${YELLOW}Running in Docker environment. App root: $APP_ROOT${NC}"
    else
        APP_ROOT="$PWD"
        DATA_DIR="$APP_ROOT/data"
        LOGS_DIR="$APP_ROOT/logs"
        TEMP_DIR="$APP_ROOT/temp_files"
        POETRY_CACHE_DIR="$HOME/.cache/pypoetry"
        echo -e "${YELLOW}Running in local environment. App root: $APP_ROOT${NC}"
    fi
fi

cd "$APP_ROOT"
echo -e "${YELLOW}Changed to directory: $(pwd)${NC}"
# Ensure PYTHONPATH includes the app directory

export PYTHONPATH="$APP_ROOT:$PYTHONPATH"
echo -e "${YELLOW}PYTHONPATH set to: $PYTHONPATH${NC}"

# Function to clean environment variables
clean_env_vars() {
    env_file="$1"
    echo -e "${YELLOW}Loading environment variables from $env_file...${NC}"

    while IFS= read -r line || [ -n "$line" ]; do
        if [[ -n "$line" && ! "$line" =~ ^# ]]; then
            cleaned_line=$(echo "$line" | tr -d '\r' | sed 's/[[:space:]]*$//')

            if [[ "$cleaned_line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                var_name="${BASH_REMATCH[1]}"
                var_value="${BASH_REMATCH[2]}"

                var_value="${var_value#\"}"
                var_value="${var_value%\"}"
                var_value="${var_value#\'}"
                var_value="${var_value%\'}"

                case "$var_name" in
                    "LOG_LEVEL")
                        var_value=$(echo "$var_value" | tr '[:upper:]' '[:lower:]')
                        ;;
                    "OPENAI_API_KEY"|"GROK_API_KEY"|"VENICE_API_KEY"|"AWS_ACCESS_KEY_ID"|"AWS_SECRET_ACCESS_KEY"|"AWS_DEFAULT_REGION"|"S3_BUCKET")
                        var_value=$(echo "$var_value" | tr -d '\r')
                        ;;
                    *)
                        var_value=$(echo "$var_value" | tr -d '\r\n' | tr '[:upper:]' '[:lower:]')
                        ;;
                esac

                # Only export if the variable isn't already set
                if [ -z "${!var_name}" ]; then
                    export "$var_name=$var_value"
                    echo "Exported: $var_name"
                fi
            fi
        fi
    done < "$env_file"
}

# Load environment variables, trying .env in multiple locations
if [ "$ENVIRONMENT_LOADED" != "true" ]; then
    if [ -f "$APP_ROOT/.env" ]; then
        clean_env_vars "$APP_ROOT/.env"
    elif [ -f ".env" ]; then
        clean_env_vars ".env"
    else
        echo "No .env file found"
    fi
    export ENVIRONMENT_LOADED=true
fi

# Set default environments if not set
export ENVIRONMENT="${ENVIRONMENT:-development}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
# Convert LOG_LEVEL to lowercase for uvicorn compatibility
LOG_LEVEL=$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')
export API_PORT="${API_PORT:-80}"
export HOST="${HOST:-0.0.0.0}"
export API_WORKERS="${API_WORKERS:-4}"
export WORKER_ID="Spawn1"
# Data scheduler settings with sensible defaults
export ENABLE_DATA_SCHEDULER="${ENABLE_DATA_SCHEDULER:-true}"
export DATA_UPDATE_INTERVAL="${DATA_UPDATE_INTERVAL:-3600}"  # Default: 1 hour in seconds
# Data retention settings
export DATA_RETENTION_DAYS="${DATA_RETENTION_DAYS:-30}"  # Default: 30 days retention

# Set Replit-specific environment variables if in Replit
if [ "$IS_REPLIT" = true ]; then
    export REPLIT_ENV="${REPLIT_ENV:-replit}"
    export FASTAPI_ENV="${FASTAPI_ENV:-production}"
    export PORT="80"  # Replit expects port 80
    export API_PORT="80"
    echo "Set Replit environment variables"
    
    # Install Replit-specific packages if needed
    if ! poetry run python -c "import replit.object_storage" &>/dev/null; then
        echo -e "${YELLOW}Installing replit-object-storage package for embedding storage...${NC}"
        poetry add replit-object-storage
    else
        echo -e "${GREEN}replit-object-storage package is already installed${NC}"
    fi
    
    # Verify Object Storage is available
    poetry run python -c "
try:
    from replit.object_storage import Client
    client = Client()
    print('✅ Replit Object Storage is available for embedding storage')
except Exception as e:
    print(f'❌ Error: Replit Object Storage is not available: {e}')
    print('Large embeddings may fail to store properly')
"
fi

# Verify required environment variables
echo "Verifying environment configuration..."
required_vars=(
    "OPENAI_API_KEY"
    "API_PORT"
    "LOG_LEVEL"
    "AWS_DEFAULT_REGION"
    "AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY"
    "S3_BUCKET"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "Verified: $var"
    else
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "Warning: Missing recommended environment variables:"
    printf '%s\n' "${missing_vars[@]}"
    # Not exiting with error, as some variables might be optional in certain environments
fi

# Print AWS configuration for debugging
echo "AWS Configuration:"
echo "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
echo "S3_BUCKET=${S3_BUCKET}"
echo "AWS_ACCESS_KEY_ID is set: $([ -n "$AWS_ACCESS_KEY_ID" ] && echo "yes" || echo "no")"
echo "AWS_SECRET_ACCESS_KEY is set: $([ -n "$AWS_SECRET_ACCESS_KEY" ] && echo "yes" || echo "no")"

# Check Replit database configuration if in Replit environment
if [ "$IS_REPLIT" = true ]; then
    echo "Replit Database Configuration:"
    echo "REPLIT_DB_URL is set: $([ -n "$REPLIT_DB_URL" ] && echo "yes" || echo "no")"
    echo "DATABASE_URL is set: $([ -n "$DATABASE_URL" ] && echo "yes" || echo "no")"
    echo "PGHOST is set: $([ -n "$PGHOST" ] && echo "yes" || echo "no")"
    echo "PGUSER is set: $([ -n "$PGUSER" ] && echo "yes" || echo "no")"
    echo "PGPASSWORD is set: $([ -n "$PGPASSWORD" ] && echo "yes" || echo "no")"
    
    # Check if we need to create a PostgreSQL database
    if [ -z "$DATABASE_URL" ] && [ -z "$PGHOST" ]; then
        echo -e "${YELLOW}No PostgreSQL database detected. You may need to create one from the Database tool in Replit.${NC}"
        echo -e "${YELLOW}Visit the Database tool in the Replit sidebar to add a PostgreSQL database.${NC}"
    fi
    
    # Check if we can access REPLIT_DB_URL for Key-Value storage
    if [ -z "$REPLIT_DB_URL" ]; then
        # For deployments, check if REPLIT_DB_URL is available in /tmp/replitdb
        if [ -f "/tmp/replitdb" ]; then
            # Read the value without printing it
            echo -e "${YELLOW}Found REPLIT_DB_URL in /tmp/replitdb for deployment${NC}"
        else
            echo -e "${YELLOW}REPLIT_DB_URL not set. Key-Value operations may fail.${NC}"
        fi
    fi
fi

# Create necessary directories with proper permissions
mkdir -p "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}"
mkdir -p "${DATA_DIR}/stratified" "${DATA_DIR}/shared"

# Export the logs directory as an environment variable
export LOGS_DIR="${LOGS_DIR}"
echo -e "${YELLOW}LOGS_DIR set to: $LOGS_DIR${NC}"

# Set proper permissions for data directories
# If running as root, change ownership to nobody:nogroup (65534:65534)
if [ "$(id -u)" = "0" ]; then
    echo -e "${YELLOW}Running as root, setting permissions for nobody:nogroup${NC}"

    # Set permissions recursively
    chown -R 65534:65534 "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}" || echo -e "${YELLOW}Warning: Could not set ownership, continuing anyway${NC}"
    chmod -R 775 "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}" || echo -e "${YELLOW}Warning: Could not set permissions, continuing anyway${NC}"
else
    # If not running as root, ensure we at least have write permissions
    echo -e "${YELLOW}Not running as root, setting permissions for current user${NC}"
    chmod -R 775 "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}" || echo -e "${YELLOW}Warning: Could not set permissions, continuing anyway${NC}"
fi

# Clean up any stale initialization markers and temporary files
echo "Cleaning up any stale initialization markers and temporary files..."

# Clean up initialization markers
rm -f "$DATA_DIR/.initialization_in_progress"
rm -f "$DATA_DIR/.initialization_state"
rm -f "$DATA_DIR/.worker.lock"
rm -f "$DATA_DIR/.worker.lock."*
rm -f "$DATA_DIR/.primary_worker"

# Clean up embedding worker lock files
echo "Cleaning up stale embedding worker lock files..."

# Add your additional cleanup tasks here if needed

# Function to start the data scheduler
start_data_scheduler() {
    local interval="${1:-3600}"
    local scheduler_pid_file="${DATA_DIR}/.scheduler_pid"
    local scheduler_log="${LOGS_DIR}/scheduler.log"
    
    # Check if scheduler is already running
    if [ -f "$scheduler_pid_file" ]; then
        local pid=$(cat "$scheduler_pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}Data scheduler is already running with PID: $pid${NC}"
            return 0
        else
            echo -e "${YELLOW}Found stale scheduler PID file, cleaning up...${NC}"
            rm -f "$scheduler_pid_file"
        fi
    fi
    
    echo -e "${YELLOW}Starting data scheduler with interval: ${interval}s${NC}"
    
    # Start the scheduler in background
    nohup python scripts/scheduled_update.py refresh --continuous --interval="$interval" > "$scheduler_log" 2>&1 &
    
    # Save the PID
    local scheduler_pid=$!
    echo "$scheduler_pid" > "$scheduler_pid_file"
    
    echo -e "${GREEN}Data scheduler started with PID: $scheduler_pid${NC}"
    echo -e "${YELLOW}Scheduler will update data every $interval seconds${NC}"
    echo -e "${YELLOW}Scheduler logs available at: $scheduler_log${NC}"
}

# Function to stop the data scheduler
stop_data_scheduler() {
    local scheduler_pid_file="${DATA_DIR}/.scheduler_pid"
    
    if [ ! -f "$scheduler_pid_file" ]; then
        echo -e "${YELLOW}Scheduler PID file not found, no scheduler to stop${NC}"
        return 0
    fi
    
    local pid=$(cat "$scheduler_pid_file")
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}Scheduler process not found, cleaning up PID file${NC}"
        rm -f "$scheduler_pid_file"
        return 0
    fi
    
    echo -e "${YELLOW}Stopping data scheduler with PID: $pid${NC}"
    kill "$pid" || true
    
    # Wait for the process to terminate
    for i in {1..5}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            break
        fi
        echo "Waiting for scheduler to terminate... ($i/5)"
        sleep 1
    done
    
    # If process is still running, force kill
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}Force killing scheduler process...${NC}"
        kill -9 "$pid" || true
    fi
    
    rm -f "$scheduler_pid_file"
    echo -e "${GREEN}Data scheduler stopped${NC}"
}

# If running tests at startup is enabled, execute tests
if [ "$RUN_TESTS_ON_STARTUP" = "true" ]; then
    echo "Running tests on startup..."
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    TEST_SCRIPT_PATH="$APP_ROOT/scripts/run_tests.sh"
    
    if [ -f "$TEST_SCRIPT_PATH" ]; then
        echo "Running tests with script: $TEST_SCRIPT_PATH"
        bash "$TEST_SCRIPT_PATH" --type="$TEST_TYPE" || {
            if [ "$ABORT_ON_TEST_FAILURE" = "true" ]; then
                echo "Tests failed and ABORT_ON_TEST_FAILURE is true, exiting"
                exit 1
            else
                echo "Tests failed but continuing anyway"
            fi
        }
    else
        echo "Test script not found at: $TEST_SCRIPT_PATH"
    fi
fi

# Initialize data if auto-check is enabled
if [ "$AUTO_CHECK_DATA" = "true" ]; then
    echo "Checking data status..."
    
    # Run data processing script to check data and initialize if needed
    python scripts/process_data.py --check || {
        echo "Data status check failed, attempting to process data..."
        
        # Force refresh if needed
        if [ "$FORCE_DATA_REFRESH" = "true" ]; then
            echo "Running data processing with force refresh..."
            python scripts/process_data.py --force-refresh || echo "Warning: Data processing failed"
        else
            echo "Running data processing without force refresh..."
            python scripts/process_data.py || echo "Warning: Data processing failed"
        fi
    }
fi

# Start the data scheduler if enabled
if [ "$ENABLE_DATA_SCHEDULER" = "true" ]; then
    echo "Data scheduler is enabled, starting..."
    start_data_scheduler "$DATA_UPDATE_INTERVAL"
else
    echo "Data scheduler is disabled, skipping..."
fi

# Register a trap to cleanup on exit
trap stop_data_scheduler EXIT

# Start the API server
echo "Starting API server on $HOST:$API_PORT with $API_WORKERS workers..."
PYTHONUNBUFFERED=1 uvicorn api.app:app --host "$HOST" --port "$API_PORT" --workers "$API_WORKERS" --log-level "$LOG_LEVEL"