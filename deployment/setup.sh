#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Read custom environment variables related to startup behavior
RUN_TESTS_ON_STARTUP=${RUN_TESTS_ON_STARTUP:-false}
TEST_TYPE=${TEST_TYPE:-all}
AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-true}
ABORT_ON_TEST_FAILURE=${ABORT_ON_TEST_FAILURE:-false}
TEST_RESULTS_DIR=${TEST_RESULTS_DIR:-"$PWD/test_results"}

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
find "$DATA_DIR" -type f -name ".worker_*_in_progress" -delete 2>/dev/null || echo "Warning: Could not clean up embedding worker lock files"
find "$DATA_DIR/stratified" -type f -name ".worker_*_in_progress" -delete 2>/dev/null || echo "Warning: Could not clean up stratified embedding worker lock files"

# Clean up any temporary files older than 7 days
if [ -d "$TEMP_DIR" ]; then
    echo "Cleaning up old temporary files..."
    find "$TEMP_DIR" -type f -mtime +7 -delete 2>/dev/null || echo "Warning: Could not clean up old temporary files"
fi

# Clean up old log files (keep last 5 days)
if [ -d "$LOGS_DIR" ]; then
    echo "Cleaning up old log files..."
    find "$LOGS_DIR" -type f -name "*.log" -mtime +5 -delete 2>/dev/null || echo "Warning: Could not clean up old log files"
    find "$LOGS_DIR" -type f -name "*.log.*" -mtime +5 -delete 2>/dev/null || echo "Warning: Could not clean up old rotated log files"
fi

# Clean up old test results (keep last 3 days)
if [ -d "$TEST_RESULTS_DIR" ]; then
    echo "Cleaning up old test results..."
    find "$TEST_RESULTS_DIR" -type f -mtime +3 -delete 2>/dev/null || echo "Warning: Could not clean up old test results"
fi

# Create logs directory
mkdir -p "$DATA_DIR/logs"
SCHEDULER_LOG="$DATA_DIR/logs/scheduler.log"

# Create setup complete marker
echo "Creating setup complete marker"
touch "$DATA_DIR/.setup_complete"
echo "Setup completed successfully"

# Define data directories
STRATIFIED_DIR="${DATA_DIR}/stratified"
COMPLETE_DATA_FILE="${DATA_DIR}/complete_data.csv"
EMBEDDINGS_DIR="${DATA_DIR}/embeddings"

# Create required directories
mkdir -p "${STRATIFIED_DIR}"
mkdir -p "${EMBEDDINGS_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${TEMP_DIR}"
touch "${SCHEDULER_LOG}"

# Check and create necessary file ownership
echo "Setting appropriate permissions on data directories..."
if [ "$(id -u)" = "0" ]; then
    # If running as root, change ownership to nobody:nogroup (65534:65534)
    chown -R 65534:65534 "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}" || true
    chmod -R 775 "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}" || true
else
    # If not running as root, ensure we at least have write permissions
    chmod -R 775 "${DATA_DIR}" "${LOGS_DIR}" "${TEMP_DIR}" "${POETRY_CACHE_DIR}" || true
fi

# Run the log fix script to ensure logs are properly consolidated
echo -e "${YELLOW}Running log fix script to consolidate logs...${NC}"
if [ -f "$APP_ROOT/scripts/fix_log_locations.py" ]; then
    poetry run python "$APP_ROOT/scripts/fix_log_locations.py"
    echo -e "${GREEN}Log consolidation completed.${NC}"
else
    echo -e "${YELLOW}Log fix script not found, skipping log consolidation.${NC}"
fi

# Function to check if data needs to be refreshed based on Chanscope approach rules
check_data_status() {
    echo "Checking data status according to Chanscope approach..."

    # If force refresh is enabled, always return need for refresh
    if [ "$FORCE_DATA_REFRESH" = "true" ]; then
        echo "Force data refresh is enabled, will refresh data regardless of current status."
        return 1
    fi

    # Check if complete_data.csv exists
    if [ ! -f "$COMPLETE_DATA_FILE" ]; then
        echo "Complete data file not found, data ingestion required."
        return 1
    fi

    # Check if embeddings exist
    if [ -z "$(ls -A $EMBEDDINGS_DIR 2>/dev/null)" ]; then
        echo "Embeddings not found, embedding generation required."
        return 1
    fi

    # Check if data is too old (based on file modification time and DATA_RETENTION_DAYS)
    if [ -n "$DATA_RETENTION_DAYS" ]; then
        # Get file modification time in seconds since epoch
        file_time=$(stat -c %Y "$COMPLETE_DATA_FILE" 2>/dev/null || stat -f %m "$COMPLETE_DATA_FILE" 2>/dev/null)
        current_time=$(date +%s)
        max_age_seconds=$((DATA_RETENTION_DAYS * 24 * 60 * 60))

        if (( current_time - file_time > max_age_seconds )); then
            echo "Complete data file is older than $DATA_RETENTION_DAYS days, refresh required."
            return 1
        fi
    fi

    echo "Data is up-to-date according to Chanscope approach rules."
    return 0
}

# Function to run tests and handle results
run_tests() {
    local test_type="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local test_log_file="${TEST_RESULTS_DIR}/chanscope_tests_${timestamp}.log"
    local test_json_file="${TEST_RESULTS_DIR}/chanscope_validation_${timestamp}.json"

    # Create test results directory if it doesn't exist
    mkdir -p "${TEST_RESULTS_DIR}"

    # Set the appropriate test argument
    if [ "$test_type" = "all" ]; then
        TEST_ARG="--all"
    else
        TEST_ARG="--${test_type}"
    fi

    echo -e "${YELLOW}Running tests with argument: $TEST_ARG${NC}"
    echo -e "${YELLOW}Test results will be saved to: $test_log_file${NC}"

    # Choose the appropriate test script based on environment
    if [ "$IS_REPLIT" = true ]; then
        TEST_SCRIPT="$APP_ROOT/scripts/run_replit_tests.sh"
    else
        TEST_SCRIPT="$APP_ROOT/scripts/run_tests.sh"
    fi

    # Run the tests
    if [ -f "$TEST_SCRIPT" ]; then
        echo -e "${YELLOW}Starting test execution at $(date) using ${TEST_SCRIPT}${NC}" | tee -a "$test_log_file"
        bash "$TEST_SCRIPT" "$TEST_ARG" 2>&1 | tee -a "$test_log_file"
        TEST_EXIT_CODE=${PIPESTATUS[0]}

        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo -e "${GREEN}All tests passed successfully!${NC}" | tee -a "$test_log_file"
            # Create a simple JSON result file
            cat > "$test_json_file" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "test_type": "$test_type",
  "status": "success",
  "exit_code": $TEST_EXIT_CODE,
  "environment": "$([ "$IS_REPLIT" = true ] && echo "replit" || echo "docker")",
  "message": "All tests passed successfully"
}
EOF
        else
            echo -e "${RED}Some tests failed with exit code: $TEST_EXIT_CODE${NC}" | tee -a "$test_log_file"
            # Create a simple JSON result file
            cat > "$test_json_file" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "test_type": "$test_type",
  "status": "failure",
  "exit_code": $TEST_EXIT_CODE,
  "environment": "$([ "$IS_REPLIT" = true ] && echo "replit" || echo "docker")",
  "message": "Tests failed with exit code $TEST_EXIT_CODE"
}
EOF
            if [ "$ABORT_ON_TEST_FAILURE" = "true" ]; then
                echo -e "${RED}Aborting startup due to test failures (ABORT_ON_TEST_FAILURE=true)${NC}" | tee -a "$test_log_file"
                exit $TEST_EXIT_CODE
            fi
        fi
    else
        echo -e "${RED}Test script not found: $TEST_SCRIPT${NC}"
        return 1
    fi

    echo -e "${YELLOW}Tests completed at $(date), continuing with normal startup...${NC}" | tee -a "$test_log_file"
    return $TEST_EXIT_CODE
}

# Run tests if enabled
if [ "$RUN_TESTS_ON_STARTUP" = "true" ] || [ "$TEST_MODE" = "true" ]; then
    echo -e "${YELLOW}Test execution on startup is enabled.${NC}"
    # Run tests in background to not block port opening
    (run_tests "$TEST_TYPE" &)
fi

# Start data initialization in the background
if [ "$AUTO_CHECK_DATA" = "true" ]; then
    if ! check_data_status; then
        echo -e "${YELLOW}Data refresh needed. Starting initial data update in background...${NC}"
        (cd "${APP_ROOT}" && python scripts/scheduled_update.py --run_once 2>&1 | tee -a "${SCHEDULER_LOG}" &)
        echo -e "${GREEN}Data initialization started in background.${NC}"
    else
        echo -e "${GREEN}Using existing data as it's up-to-date.${NC}"
    fi
fi

# Start the FastAPI application immediately
echo -e "${YELLOW}Starting FastAPI application...${NC}"
echo -e "${YELLOW}Host: $HOST, Port: $API_PORT, Workers: $API_WORKERS, Log level: $LOG_LEVEL${NC}"
echo -e "${YELLOW}Environment: $ENVIRONMENT, Replit: $IS_REPLIT${NC}"

# In Replit, use the app instance from api/app.py which has Replit-specific configurations
if [ "$IS_REPLIT" = true ]; then
    echo -e "${YELLOW}Starting with Replit-specific configuration${NC}"
    # Reduced worker count and added --reload-delay to optimize startup
    exec poetry run python -m uvicorn api.app:app --host "$HOST" --port "$API_PORT" --log-level "$LOG_LEVEL" --reload-delay 5 --workers 1 --timeout-keep-alive 30
else
    # In Docker/local, use the standard approach
    echo -e "${YELLOW}Starting with standard configuration${NC}"
    exec poetry run uvicorn api:app --host "$HOST" --port "$API_PORT" --workers "$API_WORKERS" --log-level "$LOG_LEVEL"
fi