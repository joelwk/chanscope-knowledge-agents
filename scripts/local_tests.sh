#!/bin/bash

# Local environment test configuration and setup
# This script is called by run_tests.sh when running in a local environment

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
FORCE_REFRESH="false"
AUTO_CHECK_DATA="true"
SHOW_LOGS="true"
USE_MOCK_DATA="true"
USE_MOCK_EMBEDDINGS="true"

# Function to display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --all                   Run all tests (default)"
    echo "  --data-ingestion        Run only data ingestion tests"
    echo "  --embedding             Run only embedding tests"
    echo "  --endpoints             Run only API endpoint tests"
    echo "  --chanscope-approach    Run only Chanscope approach tests"
    echo "  --force-refresh         Force data refresh before tests"
    echo "  --no-auto-check-data    Disable automatic data checking"
    echo "  --no-mock-data          Use real data instead of mock data"
    echo "  --no-mock-embeddings    Use real embeddings instead of mock embeddings"
    echo "  --no-logs               Don't show logs during test execution"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --embedding --force-refresh"
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
            --no-logs)
                export SHOW_LOGS="false"
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

# Function to create directory with error handling
create_dir() {
    local dir_path="$1"
    echo -e "${YELLOW}Creating directory: $dir_path${NC}"
    if ! mkdir -p "$dir_path" 2>/dev/null; then
        echo -e "${RED}Failed to create directory: $dir_path${NC}"
        echo -e "${YELLOW}Checking if directory already exists...${NC}"
        if [ -d "$dir_path" ]; then
            echo -e "${GREEN}Directory already exists: $dir_path${NC}"
        else
            echo -e "${RED}Cannot create or access directory: $dir_path${NC}"
            echo -e "${YELLOW}This may be due to permission issues or a read-only filesystem.${NC}"
            echo -e "${YELLOW}Please ensure you have write permissions to the parent directory.${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}Successfully created directory: $dir_path${NC}"
    fi
    return 0
}

# Function to create mock data for testing
create_mock_data() {
    echo -e "${YELLOW}Creating sample test data...${NC}"
    MOCK_DATA_FILE="${PROJECT_ROOT}/data/mock/sample_data.csv"
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

    # Copy mock data to main data directory for tests
    COMPLETE_DATA_FILE="${PROJECT_ROOT}/data/complete_data.csv"
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

# Function to find Python executable
find_python() {
    if command -v poetry &> /dev/null; then
        echo -e "${GREEN}Found Poetry, using it to run tests${NC}"
        PYTHON_CMD="poetry run python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}Using Python3${NC}"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo -e "${GREEN}Using Python${NC}"
    else
        echo -e "${RED}Error: Neither Python nor Poetry is available in your environment.${NC}"
        echo -e "${YELLOW}Please ensure Python is installed and available in your PATH.${NC}"
        exit 1
    fi
    echo "$PYTHON_CMD"
}

# Function to run tests in local environment
run_local_tests() {
    # Parse command line arguments
    parse_args "$@"

    # Determine script directory and project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

    # Change to project root
    cd "$PROJECT_ROOT"

    # Set local-specific test environment
    export TEST_MODE=true
    export USE_MOCK_DATA=${USE_MOCK_DATA:-true}
    export USE_MOCK_EMBEDDINGS=${USE_MOCK_EMBEDDINGS:-true}
    export MOCK_DATA_PATH="${PROJECT_ROOT}/data/mock"
    export FORCE_DATA_REFRESH=${FORCE_REFRESH:-false}
    export AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-true}
    export TEST_TYPE=${TEST_TYPE:-all}
    export SHOW_LOGS=${SHOW_LOGS:-true}

    # Set critical path environment variables for tests
    export ROOT_DATA_PATH="${PROJECT_ROOT}/data"
    export STRATIFIED_PATH="${PROJECT_ROOT}/data/stratified"
    export PATH_TEMP="${PROJECT_ROOT}/temp_files"
    export TEST_DATA_PATH="${PROJECT_ROOT}/data"
    export STRATIFIED_DATA_PATH="${PROJECT_ROOT}/data/stratified"

    # Create test directories with error handling
    echo -e "${YELLOW}Creating test directories...${NC}"
    create_dir "${PROJECT_ROOT}/data" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/data/stratified" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/data/shared" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/data/logs" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/data/mock" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/logs" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/temp_files" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${PROJECT_ROOT}/test_results" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"

    # Create mock data for testing
    create_mock_data

    # Create a timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_ROOT}/test_results/chanscope_tests_local_$TIMESTAMP.log"
    RESULT_FILE="${PROJECT_ROOT}/test_results/chanscope_validation_local_$TIMESTAMP.json"

    echo -e "${YELLOW}Running tests with TEST_TYPE=$TEST_TYPE, FORCE_REFRESH=$FORCE_REFRESH, AUTO_CHECK_DATA=$AUTO_CHECK_DATA${NC}"
    echo -e "${YELLOW}Mock settings: USE_MOCK_DATA=$USE_MOCK_DATA, USE_MOCK_EMBEDDINGS=$USE_MOCK_EMBEDDINGS${NC}"
    echo -e "${YELLOW}Test paths: ROOT_DATA_PATH=$ROOT_DATA_PATH, STRATIFIED_PATH=$STRATIFIED_PATH${NC}"

    # Find Python executable
    PYTHON_CMD=$(find_python)

    # Run the tests
    if [ "$SHOW_LOGS" = "true" ]; then
        # Show logs in real-time
        if [[ "$PYTHON_CMD" == "poetry run python" ]]; then
            poetry run python -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" | tee "$LOG_FILE"
        else
            $PYTHON_CMD -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" | tee "$LOG_FILE"
        fi
    else
        # Run silently and save logs to file
        echo "Running tests silently, logs will be saved to $LOG_FILE"
        if [[ "$PYTHON_CMD" == "poetry run python" ]]; then
            poetry run python -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" > "$LOG_FILE" 2>&1
        else
            $PYTHON_CMD -m pytest tests/ -v --junitxml="${PROJECT_ROOT}/test_results/test-results.xml" > "$LOG_FILE" 2>&1
        fi
    fi

    # Check exit code
    EXIT_CODE=$?

    # Display test results summary
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed successfully!${NC}"
    else
        echo -e "${RED}❌ Some tests failed with exit code: $EXIT_CODE${NC}"
        echo "Check the logs for details: $LOG_FILE"
    fi

    # Create a JSON result file
    if [ $EXIT_CODE -eq 0 ]; then
        cat > "$RESULT_FILE" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "test_type": "$TEST_TYPE",
  "status": "success",
  "exit_code": $EXIT_CODE,
  "environment": "local",
  "message": "All tests passed successfully"
}
EOF
    else
        cat > "$RESULT_FILE" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "test_type": "$TEST_TYPE",
  "status": "failure",
  "exit_code": $EXIT_CODE,
  "environment": "local",
  "message": "Tests failed with exit code $EXIT_CODE"
}
EOF
    fi

    return $EXIT_CODE
}

# If this script is run directly, execute the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_local_tests "$@"
    exit $?
fi 