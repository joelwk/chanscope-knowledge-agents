#!/bin/bash

# Replit-specific test configuration and setup
# This script is called by run_tests.sh when running in a Replit environment

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
    echo "  --debug                 Enable debug mode with verbose output"
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
            --debug)
                export DEBUG_MODE="true"
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
    
    # Debug output
    if [ "$DEBUG_MODE" = "true" ]; then
        echo -e "${YELLOW}Debug: Checking if directory exists: $dir_path${NC}"
        if [ -d "$dir_path" ]; then
            echo -e "${YELLOW}Debug: Directory already exists: $dir_path${NC}"
            ls -la "$dir_path" 2>/dev/null || echo -e "${RED}Debug: Cannot list directory contents${NC}"
        fi
    fi
    
    if ! mkdir -p "$dir_path" 2>/dev/null; then
        echo -e "${RED}Failed to create directory: $dir_path${NC}"
        echo -e "${YELLOW}Checking if directory already exists...${NC}"
        if [ -d "$dir_path" ]; then
            echo -e "${GREEN}Directory already exists: $dir_path${NC}"
            
            # Debug output
            if [ "$DEBUG_MODE" = "true" ]; then
                echo -e "${YELLOW}Debug: Directory permissions:${NC}"
                ls -la "$(dirname "$dir_path")" 2>/dev/null || echo -e "${RED}Debug: Cannot list parent directory contents${NC}"
            fi
        else
            echo -e "${RED}Cannot create or access directory: $dir_path${NC}"
            echo -e "${YELLOW}This may be due to permission issues or a read-only filesystem.${NC}"
            echo -e "${YELLOW}Please ensure you have write permissions to the parent directory.${NC}"
            
            # Debug output
            if [ "$DEBUG_MODE" = "true" ]; then
                echo -e "${YELLOW}Debug: Parent directory permissions:${NC}"
                ls -la "$(dirname "$dir_path")" 2>/dev/null || echo -e "${RED}Debug: Cannot list parent directory contents${NC}"
                echo -e "${YELLOW}Debug: Current user: $(whoami)${NC}"
                echo -e "${YELLOW}Debug: Current directory: $(pwd)${NC}"
            fi
            
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
    MOCK_DATA_FILE="${REPL_HOME}/data/mock/sample_data.csv"
    
    # Debug output
    if [ "$DEBUG_MODE" = "true" ]; then
        echo -e "${YELLOW}Debug: Mock data file path: $MOCK_DATA_FILE${NC}"
        echo -e "${YELLOW}Debug: Checking if mock data directory exists...${NC}"
        if [ -d "$(dirname "$MOCK_DATA_FILE")" ]; then
            echo -e "${YELLOW}Debug: Mock data directory exists${NC}"
            ls -la "$(dirname "$MOCK_DATA_FILE")" 2>/dev/null || echo -e "${RED}Debug: Cannot list mock data directory contents${NC}"
        else
            echo -e "${RED}Debug: Mock data directory does not exist!${NC}"
        fi
    fi
    
    if [ ! -f "$MOCK_DATA_FILE" ]; then
        cat > "$MOCK_DATA_FILE" << EOF
thread_id,posted_date_time,text_clean,posted_comment
1001,2025-01-01 12:00:00,This is a test post for embedding generation,Original comment 1
1002,2025-01-01 12:05:00,Another test post with different content,Original comment 2
1003,2025-01-01 12:10:00,Third test post for validation purposes,Original comment 3
1004,2025-01-01 12:15:00,Fourth test post with unique content,Original comment 4
1005,2025-01-01 12:20:00,Fifth test post for comprehensive testing,Original comment 5
EOF
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Sample test data created successfully${NC}"
        else
            echo -e "${RED}Failed to create sample test data${NC}"
            
            # Debug output
            if [ "$DEBUG_MODE" = "true" ]; then
                echo -e "${YELLOW}Debug: Checking file permissions:${NC}"
                ls -la "$(dirname "$MOCK_DATA_FILE")" 2>/dev/null || echo -e "${RED}Debug: Cannot list directory contents${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Sample test data already exists, skipping creation${NC}"
    fi

    # Copy mock data to main data directory for tests
    COMPLETE_DATA_FILE="${REPL_HOME}/data/complete_data.csv"
    if [ ! -f "$COMPLETE_DATA_FILE" ]; then
        if [ -f "$MOCK_DATA_FILE" ]; then
            cp "$MOCK_DATA_FILE" "$COMPLETE_DATA_FILE"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Copied mock data to complete_data.csv${NC}"
            else
                echo -e "${RED}Failed to copy mock data to complete_data.csv${NC}"
                
                # Debug output
                if [ "$DEBUG_MODE" = "true" ]; then
                    echo -e "${YELLOW}Debug: Checking file permissions:${NC}"
                    ls -la "$(dirname "$COMPLETE_DATA_FILE")" 2>/dev/null || echo -e "${RED}Debug: Cannot list directory contents${NC}"
                fi
            fi
        else
            echo -e "${RED}Mock data file not found, cannot copy to complete_data.csv${NC}"
        fi
    else
        echo -e "${YELLOW}complete_data.csv already exists, skipping copy${NC}"
    fi
}

# Function to install required dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies using Poetry...${NC}"
    
    # Check if we're in a Replit environment
    if [ -n "$REPL_ID" ] || [ "$REPLIT_ENV" = "replit" ]; then
        echo -e "${YELLOW}Using Poetry in Replit environment...${NC}"
        
        # Check if Poetry is available
        if ! command -v poetry &> /dev/null; then
            echo -e "${RED}Poetry not found. Ensure Poetry is installed in your Replit environment.${NC}"
            return 1
        fi
        
        # Initialize Poetry environment if needed
        echo -e "${YELLOW}Initializing Poetry environment...${NC}"
        
        # Ensure pyproject.toml exists
        if [ ! -f "pyproject.toml" ]; then
            echo -e "${YELLOW}Creating pyproject.toml...${NC}"
            cat > pyproject.toml << EOF
[tool.poetry]
name = "chanscope-tests"
version = "0.1.0"
description = "Chanscope testing environment"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
pytest = "^7.0.0"
pytest-asyncio = "^0.18.0"
fastapi = "^0.95.0"
uvicorn = "^0.21.0"
httpx = "^0.24.0"
python-dotenv = "^1.0.0"
openai = "^0.27.0"
tenacity = "^8.2.0"
tiktoken = "^0.3.0"
filelock = "^3.10.0"
numpy = "^1.24.0"
pandas = "^2.0.0"
boto3 = "^1.26.0"
pydantic = "^1.10.0"
aiohttp = "^3.8.0"
pytz = "^2023.3"
PyYAML = "^6.0"

[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
EOF
        fi
        
        # Explicitly set Python version for Poetry
        echo -e "${YELLOW}Setting Python version for Poetry...${NC}"
        if ! poetry env use python3; then
            echo -e "${RED}Failed to set Python version for Poetry.${NC}"
            return 1
        fi
        
        # Install dependencies from pyproject.toml
        echo -e "${YELLOW}Installing dependencies with Poetry...${NC}"
        if ! poetry install --no-interaction; then
            echo -e "${RED}Poetry failed to install dependencies.${NC}"
            
            # Try with sync option if install fails
            echo -e "${YELLOW}Retrying with poetry lock and sync...${NC}"
            if ! poetry lock && poetry install --sync --no-interaction; then
                echo -e "${RED}Poetry sync failed to install dependencies.${NC}"
                return 1
            fi
        fi
        
        # Verify core installations
        echo -e "${YELLOW}Verifying core installations...${NC}"
        if poetry run python -c "import pytest, fastapi, uvicorn, pandas, numpy" 2>/dev/null; then
            echo -e "${GREEN}All core dependencies installed successfully${NC}"
            return 0
        else
            echo -e "${RED}Some core dependencies failed to install correctly${NC}"
            
            # Debug output
            if [ "$DEBUG_MODE" = "true" ]; then
                echo -e "${YELLOW}Debug: Listing installed packages in Poetry environment...${NC}"
                poetry run pip list
            fi
            
            return 1
        fi
    else
        # Not in Replit environment, use regular pip install
        echo -e "${YELLOW}Not in Replit environment, using pip...${NC}"
        python3 -m pip install --no-cache-dir pytest pytest-asyncio fastapi uvicorn httpx \
            python-dotenv openai tenacity tiktoken filelock numpy pandas \
            boto3 pydantic aiohttp pytz PyYAML
        return $?
    fi
}

# Function to run tests in Replit environment
run_replit_tests() {
    # Parse command line arguments
    parse_args "$@"

    # Determine script directory and project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

    # Change to project root
    cd "$PROJECT_ROOT"

    # Debug output
    if [ "$DEBUG_MODE" = "true" ]; then
        echo -e "${YELLOW}Debug: Script directory: $SCRIPT_DIR${NC}"
        echo -e "${YELLOW}Debug: Project root: $PROJECT_ROOT${NC}"
        echo -e "${YELLOW}Debug: Current directory: $(pwd)${NC}"
    fi

    # Simple check for Replit environment
    if [ -n "$REPL_ID" ] || [ "$REPLIT_ENV" = "replit" ]; then
        IS_REPLIT=true
        echo -e "${YELLOW}Detected Replit environment${NC}"
        
        # Debug output
        if [ "$DEBUG_MODE" = "true" ]; then
            echo -e "${YELLOW}Debug: REPL_ID: ${REPL_ID:-not set}${NC}"
            echo -e "${YELLOW}Debug: REPLIT_ENV: ${REPLIT_ENV:-not set}${NC}"
        fi
    else
        IS_REPLIT=false
        echo -e "${YELLOW}Not in Replit environment, assuming local execution${NC}"
    fi

    # Ensure REPL_HOME is set
    if [ -z "$REPL_HOME" ]; then
        echo -e "${YELLOW}REPL_HOME is not set, using PROJECT_ROOT instead${NC}"
        REPL_HOME="$PROJECT_ROOT"
    fi
    
    # Debug output
    if [ "$DEBUG_MODE" = "true" ]; then
        echo -e "${YELLOW}Debug: REPL_HOME: $REPL_HOME${NC}"
    fi

    # Set Replit-specific test environment
    export TEST_MODE=true
    export USE_MOCK_DATA=${USE_MOCK_DATA:-true}
    export USE_MOCK_EMBEDDINGS=${USE_MOCK_EMBEDDINGS:-true}
    export MOCK_DATA_PATH="${REPL_HOME}/data/mock"
    export FORCE_DATA_REFRESH=${FORCE_REFRESH:-false}
    export AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-true}
    export TEST_TYPE=${TEST_TYPE:-all}
    export SHOW_LOGS=${SHOW_LOGS:-true}
    export DEBUG_MODE=${DEBUG_MODE:-false}

    # Set critical path environment variables for tests
    export ROOT_DATA_PATH="${REPL_HOME}/data"
    export STRATIFIED_PATH="${REPL_HOME}/data/stratified"
    export PATH_TEMP="${REPL_HOME}/temp_files"
    export TEST_DATA_PATH="${REPL_HOME}/data"
    export STRATIFIED_DATA_PATH="${REPL_HOME}/data/stratified"

    # Debug output
    if [ "$DEBUG_MODE" = "true" ]; then
        echo -e "${YELLOW}Debug: Environment variables set:${NC}"
        echo -e "${YELLOW}Debug: TEST_MODE: $TEST_MODE${NC}"
        echo -e "${YELLOW}Debug: USE_MOCK_DATA: $USE_MOCK_DATA${NC}"
        echo -e "${YELLOW}Debug: USE_MOCK_EMBEDDINGS: $USE_MOCK_EMBEDDINGS${NC}"
        echo -e "${YELLOW}Debug: MOCK_DATA_PATH: $MOCK_DATA_PATH${NC}"
        echo -e "${YELLOW}Debug: FORCE_DATA_REFRESH: $FORCE_DATA_REFRESH${NC}"
        echo -e "${YELLOW}Debug: AUTO_CHECK_DATA: $AUTO_CHECK_DATA${NC}"
        echo -e "${YELLOW}Debug: TEST_TYPE: $TEST_TYPE${NC}"
        echo -e "${YELLOW}Debug: SHOW_LOGS: $SHOW_LOGS${NC}"
        echo -e "${YELLOW}Debug: DEBUG_MODE: $DEBUG_MODE${NC}"
        echo -e "${YELLOW}Debug: ROOT_DATA_PATH: $ROOT_DATA_PATH${NC}"
        echo -e "${YELLOW}Debug: STRATIFIED_PATH: $STRATIFIED_PATH${NC}"
        echo -e "${YELLOW}Debug: PATH_TEMP: $PATH_TEMP${NC}"
        echo -e "${YELLOW}Debug: TEST_DATA_PATH: $TEST_DATA_PATH${NC}"
        echo -e "${YELLOW}Debug: STRATIFIED_DATA_PATH: $STRATIFIED_DATA_PATH${NC}"
    fi

    # Create test directories with error handling
    echo -e "${YELLOW}Creating test directories...${NC}"
    create_dir "${REPL_HOME}/data" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/data/stratified" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/data/shared" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/data/logs" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/data/mock" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/logs" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/temp_files" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"
    create_dir "${REPL_HOME}/test_results" || echo -e "${YELLOW}Continuing despite directory creation issue...${NC}"

    # Create mock data for testing
    create_mock_data

    # Run tests with smaller batches for Replit's limited resources
    export EMBEDDING_BATCH_SIZE=5
    export CHUNK_BATCH_SIZE=5
    export PROCESSING_CHUNK_SIZE=1000

    # Create a timestamp for log files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${REPL_HOME}/test_results/chanscope_tests_replit_$TIMESTAMP.log"
    RESULT_FILE="${REPL_HOME}/test_results/chanscope_validation_replit_$TIMESTAMP.json"

    echo -e "${YELLOW}Running tests with TEST_TYPE=$TEST_TYPE, FORCE_REFRESH=$FORCE_REFRESH, AUTO_CHECK_DATA=$AUTO_CHECK_DATA${NC}"
    echo -e "${YELLOW}Mock settings: USE_MOCK_DATA=$USE_MOCK_DATA, USE_MOCK_EMBEDDINGS=$USE_MOCK_EMBEDDINGS${NC}"
    echo -e "${YELLOW}Test paths: ROOT_DATA_PATH=$ROOT_DATA_PATH, STRATIFIED_PATH=$STRATIFIED_PATH${NC}"
    echo -e "${YELLOW}Optimization settings: EMBEDDING_BATCH_SIZE=$EMBEDDING_BATCH_SIZE, CHUNK_BATCH_SIZE=$CHUNK_BATCH_SIZE${NC}"
    echo -e "${YELLOW}Processing settings: PROCESSING_CHUNK_SIZE=$PROCESSING_CHUNK_SIZE, STRATIFICATION_CHUNK_SIZE=$STRATIFICATION_CHUNK_SIZE${NC}"
    echo -e "${YELLOW}Worker settings: MAX_WORKERS=$MAX_WORKERS, SAMPLE_SIZE=$SAMPLE_SIZE${NC}"

    # Check if Poetry is available
    if command -v poetry &> /dev/null; then
        PYTHON_CMD="poetry run python"
        echo -e "${GREEN}Using Poetry's Python environment${NC}"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}Using Python3${NC}"
    else
        echo -e "${RED}Error: Neither Poetry nor Python3 is available in your environment.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Using $PYTHON_CMD for tests${NC}"

    # Install dependencies if needed
    if ! poetry run python -c "import pytest" &> /dev/null; then
        echo -e "${YELLOW}pytest not found, installing required dependencies...${NC}"
        install_dependencies
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install dependencies${NC}"
            return 1
        fi
    fi

    # Run the tests
    if [ "$SHOW_LOGS" = "true" ]; then
        echo -e "${YELLOW}Running tests...${NC}"
        if command -v poetry &> /dev/null; then
            poetry run pytest tests/ -v --junitxml="${REPL_HOME}/test_results/test-results.xml" | tee "$LOG_FILE"
            TEST_EXIT_CODE=${PIPESTATUS[0]}
        else
            python3 -m pytest tests/ -v --junitxml="${REPL_HOME}/test_results/test-results.xml" | tee "$LOG_FILE"
            TEST_EXIT_CODE=${PIPESTATUS[0]}
        fi
    else
        echo -e "${YELLOW}Running tests (logs saved to $LOG_FILE)...${NC}"
        if command -v poetry &> /dev/null; then
            poetry run pytest tests/ -v --junitxml="${REPL_HOME}/test_results/test-results.xml" > "$LOG_FILE" 2>&1
            TEST_EXIT_CODE=$?
        else
            python3 -m pytest tests/ -v --junitxml="${REPL_HOME}/test_results/test-results.xml" > "$LOG_FILE" 2>&1
            TEST_EXIT_CODE=$?
        fi
    fi

    # Report test results
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}All tests passed successfully!${NC}"
    else
        echo -e "${RED}Some tests failed. Check $LOG_FILE for details.${NC}"
    fi

    return $TEST_EXIT_CODE
}

# If this script is run directly, execute the function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_replit_tests "$@"
    exit $?
fi 