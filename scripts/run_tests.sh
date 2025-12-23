#!/bin/bash

# Unified test runner script for Chanscope
# Auto-detects environment and runs appropriate tests with consistent interface

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
ENVIRONMENT="auto"
DEBUG_MODE="false"
TEST_TYPE="all"
FORCE_REFRESH="false"
AUTO_CHECK_DATA="true"
CLEAN_VOLUMES="false"
SHOW_LOGS="true"
USE_MOCK_DATA="true"
USE_MOCK_EMBEDDINGS="true"
PYTHON_BIN=""
LOG_TAIL_LINES=200

# Resolve Python interpreter for local/replit tests (keeps CI/pip in sync)
resolve_python_bin() {
    if [ -n "$PYTHON_BIN" ]; then
        return
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo -e "${RED}Error: Python interpreter not found in PATH${NC}"
        exit 1
    fi
}

# Function to display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all                   Run all tests (default)"
    echo "  --data-ingestion        Run only data ingestion tests"
    echo "  --embedding             Run only embedding tests"
    echo "  --endpoints             Run only API endpoint tests"
    echo "  --chanscope-approach    Run only Chanscope approach validation tests"
    echo "  --force-refresh         Force data refresh before tests"
    echo "  --no-auto-check-data    Disable automatic data checking"
    echo "  --no-mock-data          Use real data instead of mock data (local/Replit only)"
    echo "  --no-mock-embeddings    Use real embeddings instead of mock embeddings (local/Replit only)"
    echo "  --clean                 Clean test volumes before running tests (Docker only)"
    echo "  --no-logs               Don't show logs during test execution"
    echo "  --log-tail-lines=<n>    Number of log lines to print on failure when using --no-logs (default: $LOG_TAIL_LINES)"
    echo "  --debug                 Enable debug mode with verbose output"
    echo "  --env=<environment>     Specify environment: local, docker, or replit"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --embedding --force-refresh"
    echo "  $0 --env=docker --clean"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                export TEST_TYPE="all"
                shift
                ;;
            --data-ingestion)
                export TEST_TYPE="data-ingestion"
                shift
                ;;
            --embedding)
                export TEST_TYPE="embedding"
                shift
                ;;
            --endpoints)
                export TEST_TYPE="endpoints"
                shift
                ;;
            --chanscope-approach)
                export TEST_TYPE="chanscope-approach"
                shift
                ;;
            --force-refresh)
                export FORCE_REFRESH="true"
                shift
                ;;
            --auto-check-data)
                export AUTO_CHECK_DATA="true"
                shift
                ;;
            --no-auto-check-data)
                export AUTO_CHECK_DATA="false"
                shift
                ;;
            --no-mock-data)
                export USE_MOCK_DATA="false"
                shift
                ;;
            --no-mock-embeddings)
                export USE_MOCK_EMBEDDINGS="false"
                shift
                ;;
            --clean)
                export CLEAN_VOLUMES="true"
                shift
                ;;
            --no-logs)
                export SHOW_LOGS="false"
                shift
                ;;
            --log-tail-lines=*)
                LOG_TAIL_LINES="${1#*=}"
                shift
                ;;
            --debug)
                export DEBUG_MODE="true"
                shift
                ;;
            --env=*)
                ENVIRONMENT="${1#*=}"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Print tail of log file on failure when running silently
print_log_tail_on_failure() {
    local exit_code=$1
    local log_file=$2
    local lines=${3:-$LOG_TAIL_LINES}

    if [ "$SHOW_LOGS" = "false" ] && [ "$exit_code" -ne 0 ] && [ -f "$log_file" ]; then
        echo -e "${YELLOW}--- Last ${lines} lines of log (${log_file}) ---${NC}"
        tail -n "$lines" "$log_file" || true
        echo -e "${YELLOW}--- End log excerpt ---${NC}"
    fi
}

# Auto-detect environment if not specified
detect_environment() {
    if [ "$ENVIRONMENT" = "auto" ]; then
        # Check for Docker environment
        if [ -f "/.dockerenv" ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
            ENVIRONMENT="docker"
            echo -e "${YELLOW}Auto-detected Docker environment${NC}"
        # Check for Replit environment - enhanced detection
        elif [ -n "$REPL_ID" ] || [ -n "$REPL_SLUG" ] || [ -n "$REPL_OWNER" ] || [ "$REPLIT_ENV" = "replit" ] || [ -d "/home/runner" ]; then
            ENVIRONMENT="replit"
            echo -e "${YELLOW}Auto-detected Replit environment${NC}"
        # Default to local environment
        else
            ENVIRONMENT="local"
            echo -e "${YELLOW}Auto-detected local environment${NC}"
        fi
    fi

    # Allow manual override via environment variable
    if [ -n "$FORCE_ENVIRONMENT" ]; then
        echo -e "${YELLOW}Environment manually overridden to: $FORCE_ENVIRONMENT${NC}"
        ENVIRONMENT="$FORCE_ENVIRONMENT"
    fi

    # Validate environment
    if [ "$ENVIRONMENT" != "local" ] && [ "$ENVIRONMENT" != "docker" ] && [ "$ENVIRONMENT" != "replit" ]; then
        echo -e "${RED}Error: Invalid environment specified: $ENVIRONMENT${NC}"
        echo -e "${YELLOW}Valid environments are: local, docker, replit${NC}"
        exit 1
    fi
}

# Create directories needed for testing (with error handling)
create_test_directories() {
    local env_type=$1
    echo -e "${YELLOW}Creating test directories for $env_type environment...${NC}"
    
    if [ "$env_type" = "docker" ]; then
        # Docker directories are managed by docker-compose
        return 0
    fi
    
    # Common directories for local and Replit environments
    mkdir -p "${PROJECT_ROOT}/data" || echo -e "${YELLOW}Warning: Failed to create data directory${NC}"
    mkdir -p "${PROJECT_ROOT}/data/stratified" || echo -e "${YELLOW}Warning: Failed to create stratified directory${NC}"
    mkdir -p "${PROJECT_ROOT}/data/shared" || echo -e "${YELLOW}Warning: Failed to create shared directory${NC}"
    mkdir -p "${PROJECT_ROOT}/data/logs" || echo -e "${YELLOW}Warning: Failed to create data logs directory${NC}"
    mkdir -p "${PROJECT_ROOT}/data/mock" || echo -e "${YELLOW}Warning: Failed to create mock data directory${NC}"
    mkdir -p "${PROJECT_ROOT}/logs" || echo -e "${YELLOW}Warning: Failed to create logs directory${NC}"
    mkdir -p "${PROJECT_ROOT}/temp_files" || echo -e "${YELLOW}Warning: Failed to create temp files directory${NC}"
    mkdir -p "${PROJECT_ROOT}/test_results" || echo -e "${YELLOW}Warning: Failed to create test results directory${NC}"
    
    # Create mock data if needed and USE_MOCK_DATA is true
    if [ "$USE_MOCK_DATA" = "true" ]; then
        create_mock_data
    fi
}

# Create mock data for testing
create_mock_data() {
    echo -e "${YELLOW}Creating sample test data...${NC}"
    MOCK_DATA_FILE="${PROJECT_ROOT}/data/mock/sample_data.csv"
    COMPLETE_DATA_FILE="${PROJECT_ROOT}/data/complete_data.csv"
    
    if [ ! -f "$MOCK_DATA_FILE" ]; then
        cat > "$MOCK_DATA_FILE" << EOF
thread_id,posted_date_time,text_clean,posted_comment
1001,2025-01-01 12:00:00,This is a test post for embedding generation,Original comment 1
1002,2025-01-01 12:05:00,Another test post with different content,Original comment 2
1003,2025-01-01 12:10:00,Third test post for validation purposes,Original comment 3
1004,2025-01-01 12:15:00,Fourth test post with unique content,Original comment 4
1005,2025-01-01 12:20:00,Fifth test post for comprehensive testing,Original comment 5
EOF
        echo -e "${GREEN}Sample test data created successfully${NC}"
    else
        echo -e "${YELLOW}Sample test data already exists, skipping creation${NC}"
    fi

    # Copy mock data to main data directory for tests if needed
    if [ ! -f "$COMPLETE_DATA_FILE" ]; then
        if [ -f "$MOCK_DATA_FILE" ]; then
            cp "$MOCK_DATA_FILE" "$COMPLETE_DATA_FILE"
            echo -e "${GREEN}Copied mock data to complete_data.csv${NC}"
        else
            echo -e "${RED}Mock data file not found, cannot copy to complete_data.csv${NC}"
        fi
    else
        echo -e "${YELLOW}complete_data.csv already exists, skipping copy${NC}"
    fi
}

# Run Docker Tests
run_docker_tests() {
    echo -e "${YELLOW}Running tests in Docker environment...${NC}"
    
    # Create a timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_ROOT}/test_results/chanscope_tests_docker_$TIMESTAMP.log"
    
    # Clean test volumes if requested
    if [ "$CLEAN_VOLUMES" = "true" ]; then
        echo -e "${YELLOW}Cleaning test volumes...${NC}"
        docker-compose -f deployment/docker-compose.test.yml down -v
        echo -e "${GREEN}Test volumes cleaned.${NC}"
    fi
    
    # Build the test image
    echo -e "${YELLOW}Building test Docker image...${NC}"
    docker-compose -f deployment/docker-compose.test.yml build
    
    # Run the tests with appropriate environment variables
    echo -e "${YELLOW}Running Docker tests with TEST_TYPE=$TEST_TYPE, FORCE_REFRESH=$FORCE_REFRESH, AUTO_CHECK_DATA=$AUTO_CHECK_DATA${NC}"
    
    # Special handling for Chanscope approach tests
    if [ "$TEST_TYPE" = "chanscope-approach" ]; then
        # For Chanscope approach tests, run validation script directly
        docker-compose -f deployment/docker-compose.test.yml run --rm \
            -e TEST_MODE=true \
            -e USE_MOCK_DATA="$USE_MOCK_DATA" \
            -e USE_MOCK_EMBEDDINGS="$USE_MOCK_EMBEDDINGS" \
            -e FORCE_DATA_REFRESH="$FORCE_REFRESH" \
            -e AUTO_CHECK_DATA="$AUTO_CHECK_DATA" \
            test-runner python scripts/validate_chanscope_approach.py \
            | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        # For regular tests, use the Docker test command
        if [ "$SHOW_LOGS" = "true" ]; then
            # Show logs in real-time
            docker-compose -f deployment/docker-compose.test.yml run --rm \
                -e TEST_MODE=true \
                -e USE_MOCK_DATA="$USE_MOCK_DATA" \
                -e USE_MOCK_EMBEDDINGS="$USE_MOCK_EMBEDDINGS" \
                -e TEST_TYPE="$TEST_TYPE" \
                -e FORCE_DATA_REFRESH="$FORCE_REFRESH" \
                -e AUTO_CHECK_DATA="$AUTO_CHECK_DATA" \
                test-runner | tee "$LOG_FILE"
            EXIT_CODE=${PIPESTATUS[0]}
        else
            # Run silently and save logs to file
            echo -e "${YELLOW}Running tests silently, logs will be saved to $LOG_FILE${NC}"
            docker-compose -f deployment/docker-compose.test.yml run --rm \
                -e TEST_MODE=true \
                -e USE_MOCK_DATA="$USE_MOCK_DATA" \
                -e USE_MOCK_EMBEDDINGS="$USE_MOCK_EMBEDDINGS" \
                -e TEST_TYPE="$TEST_TYPE" \
                -e FORCE_DATA_REFRESH="$FORCE_REFRESH" \
                -e AUTO_CHECK_DATA="$AUTO_CHECK_DATA" \
                test-runner > "$LOG_FILE" 2>&1
            EXIT_CODE=$?
        fi
    fi

    # Display test results summary
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    else
        echo -e "${RED}❌ Some tests failed with exit code: $EXIT_CODE${NC}"
        echo -e "${YELLOW}Check the logs for details: $LOG_FILE${NC}"
        print_log_tail_on_failure "$EXIT_CODE" "$LOG_FILE"
    fi

    return $EXIT_CODE
}

# Run Local Tests
run_local_tests() {
    echo -e "${YELLOW}Running tests in local environment...${NC}"
    
    # Create a timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_ROOT}/test_results/chanscope_tests_local_$TIMESTAMP.log"
    
    # Set local-specific test environment
    export TEST_MODE=true
    export USE_MOCK_DATA="${USE_MOCK_DATA:-true}"
    export USE_MOCK_EMBEDDINGS="${USE_MOCK_EMBEDDINGS:-true}"
    export FORCE_DATA_REFRESH=${FORCE_REFRESH:-false}
    export AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-true}
    export FORCE_ENVIRONMENT="local"
    
    # Set critical path environment variables for tests
    export ROOT_DATA_PATH="${PROJECT_ROOT}/data"
    export STRATIFIED_PATH="${PROJECT_ROOT}/data/stratified"
    export PATH_TEMP="${PROJECT_ROOT}/temp_files"
    export TEST_DATA_PATH="${PROJECT_ROOT}/data"
    export STRATIFIED_DATA_PATH="${PROJECT_ROOT}/data/stratified"
    # Force POSIX temp dir to avoid Windows-mounted temp issues
    export TMPDIR="/tmp"
    export TEMP="/tmp"
    export TMP="/tmp"
    
    resolve_python_bin
    echo -e "${YELLOW}Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))${NC}"

    # Run regular pytest tests
    if [ "$SHOW_LOGS" = "true" ]; then
        # Show logs in real-time
        set +e
        "$PYTHON_BIN" -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e
    else
        # Run silently and save logs to file
        echo -e "${YELLOW}Running tests silently, logs will be saved to $LOG_FILE${NC}"
        set +e
        "$PYTHON_BIN" -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" > "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        set -e
    fi

    # Display test results summary
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    else
        echo -e "${RED}❌ Some tests failed with exit code: $EXIT_CODE${NC}"
        echo -e "${YELLOW}Check the logs for details: $LOG_FILE${NC}"
        print_log_tail_on_failure "$EXIT_CODE" "$LOG_FILE"
    fi

    return $EXIT_CODE
}

# Run Replit Tests
run_replit_tests() {
    echo -e "${YELLOW}Running tests in Replit environment...${NC}"
    
    # Create a timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_ROOT}/test_results/chanscope_tests_replit_$TIMESTAMP.log"
    
    # Set Replit-specific environment variables
    export REPLIT_ENV="replit"
    export REPL_ID="${REPL_ID:-replit_test_run}"
    export TEST_MODE=true
    export USE_MOCK_DATA="${USE_MOCK_DATA:-true}"
    export USE_MOCK_EMBEDDINGS="${USE_MOCK_EMBEDDINGS:-true}"
    export FORCE_DATA_REFRESH=${FORCE_REFRESH:-false}
    export AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-true}
    export FORCE_ENVIRONMENT="replit"
    
    # Set resource-optimized settings for Replit
    export EMBEDDING_BATCH_SIZE=5
    export CHUNK_BATCH_SIZE=5
    export PROCESSING_CHUNK_SIZE=1000
    export MAX_WORKERS=1
    
    # Set critical path environment variables for tests
    export ROOT_DATA_PATH="${PROJECT_ROOT}/data"
    export STRATIFIED_PATH="${PROJECT_ROOT}/data/stratified"
    export PATH_TEMP="${PROJECT_ROOT}/temp_files"
    export TEST_DATA_PATH="${PROJECT_ROOT}/data"
    export STRATIFIED_DATA_PATH="${PROJECT_ROOT}/data/stratified"
    # Force POSIX temp dir to avoid Windows-mounted temp issues
    export TMPDIR="/tmp"
    export TEMP="/tmp"
    export TMP="/tmp"
    
    # Run Replit setup script if available
    if [ -f "${SCRIPT_DIR}/replit_setup.sh" ]; then
        echo -e "${YELLOW}Running Replit setup script...${NC}"
        bash "${SCRIPT_DIR}/replit_setup.sh"
    fi
    
    resolve_python_bin
    echo -e "${YELLOW}Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))${NC}"

    # Run regular pytest tests
    if [ "$SHOW_LOGS" = "true" ]; then
        # Show logs in real-time
        set +e
        "$PYTHON_BIN" -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e
    else
        # Run silently and save logs to file
        echo -e "${YELLOW}Running tests silently, logs will be saved to $LOG_FILE${NC}"
        set +e
        "$PYTHON_BIN" -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" > "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        set -e
    fi

    # Display test results summary
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    else
        echo -e "${RED}❌ Some tests failed with exit code: $EXIT_CODE${NC}"
        echo -e "${YELLOW}Check the logs for details: $LOG_FILE${NC}"
        print_log_tail_on_failure "$EXIT_CODE" "$LOG_FILE"
    fi

    return $EXIT_CODE
}

# Main function
main() {
    # Parse command line arguments
    parse_args "$@"
    
    # Detect environment
    detect_environment
    
    echo -e "${YELLOW}Running tests in $ENVIRONMENT environment${NC}"
    
    # Create test directories
    create_test_directories "$ENVIRONMENT"
    
    # Run tests based on environment
    case "$ENVIRONMENT" in
        local)
            run_local_tests
            EXIT_CODE=$?
            ;;
        docker)
            run_docker_tests
            EXIT_CODE=$?
            ;;
        replit)
            run_replit_tests
            EXIT_CODE=$?
            ;;
    esac
    
    return $EXIT_CODE
}

# Run main function with all arguments
main "$@"
exit $? 
