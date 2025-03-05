#!/bin/bash

# Main test runner script for Chanscope
# This script detects the current environment and calls the appropriate environment-specific test script

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
    echo "  --no-mock-data          Use real data instead of mock data (local/Replit only)"
    echo "  --no-mock-embeddings    Use real embeddings instead of mock embeddings (local/Replit only)"
    echo "  --clean                 Clean test volumes before running tests (Docker only)"
    echo "  --no-logs               Don't show logs during test execution"
    echo "  --debug                 Enable debug mode with verbose output"
    echo "  --env=<environment>     Specify environment: local, docker, or replit"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --embedding --force-refresh"
    echo "  $0 --env=docker --clean"
}

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default environment is auto-detect
ENVIRONMENT="auto"
DEBUG_MODE="false"

# Parse command line arguments to extract environment and debug flag
for arg in "$@"; do
    if [[ $arg == --env=* ]]; then
        ENVIRONMENT="${arg#*=}"
    elif [[ $arg == --debug ]]; then
        DEBUG_MODE="true"
        echo -e "${YELLOW}Debug mode enabled - verbose output will be shown${NC}"
    fi
done

# Auto-detect environment if not specified
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

# Debug output for environment detection
if [ "$DEBUG_MODE" = "true" ]; then
    echo -e "${YELLOW}Debug: Environment detection details:${NC}"
    echo "REPL_ID=${REPL_ID:-not set}"
    echo "REPL_SLUG=${REPL_SLUG:-not set}"
    echo "REPL_OWNER=${REPL_OWNER:-not set}"
    echo "REPLIT_ENV=${REPLIT_ENV:-not set}"
    echo "FORCE_ENVIRONMENT=${FORCE_ENVIRONMENT:-not set}"
    echo "Docker check: $([ -f "/.dockerenv" ] && echo "true" || echo "false")"
    echo "Home directory: $HOME"
    echo "Current directory: $(pwd)"
    echo "Project root: $PROJECT_ROOT"
    echo "Script directory: $SCRIPT_DIR"
fi

echo -e "${YELLOW}Running tests in $ENVIRONMENT environment${NC}"

# Load environment-specific configuration from .env.test if it exists
if [ -f "$PROJECT_ROOT/.env.test" ]; then
    echo -e "${YELLOW}Loading configuration from .env.test${NC}"
    
    # Create a temporary file for environment variables
    TEMP_ENV_FILE=$(mktemp)
    
    # Extract shared configuration (lines before any section header)
    grep -B1000 "^\[" "$PROJECT_ROOT/.env.test" | grep -v "^\[" | grep -v "^$" | grep -v "^#" > "$TEMP_ENV_FILE"
    
    # Extract environment-specific configuration
    grep -A1000 "^\[$ENVIRONMENT\]" "$PROJECT_ROOT/.env.test" | grep -v "^\[" | grep -v "^$" | grep -v "^#" >> "$TEMP_ENV_FILE"
    
    # Export variables from the temporary file
    set -a
    source "$TEMP_ENV_FILE"
    set +a
    
    # Clean up
    rm "$TEMP_ENV_FILE"
    
    # Debug output for loaded environment variables
    if [ "$DEBUG_MODE" = "true" ]; then
        echo -e "${YELLOW}Debug: Loaded environment variables from .env.test:${NC}"
        echo "TEST_MODE=${TEST_MODE:-not set}"
        echo "REPLIT_ENV=${REPLIT_ENV:-not set}"
        echo "REPL_ID=${REPL_ID:-not set}"
        echo "USE_MOCK_DATA=${USE_MOCK_DATA:-not set}"
        echo "USE_MOCK_EMBEDDINGS=${USE_MOCK_EMBEDDINGS:-not set}"
    fi
fi

# Export debug mode for environment-specific scripts
export DEBUG_MODE="$DEBUG_MODE"

# Filter out the --env option from arguments
FILTERED_ARGS=()
for arg in "$@"; do
    if [[ ! $arg == --env=* ]]; then
        FILTERED_ARGS+=("$arg")
    fi
done

# Call the appropriate environment-specific test script
case "$ENVIRONMENT" in
    local)
        # Source the local test script and run tests
        source "$SCRIPT_DIR/local_tests.sh"
        run_local_tests "${FILTERED_ARGS[@]}"
        EXIT_CODE=$?
        ;;
    docker)
        # Source the Docker test script and run tests
        source "$SCRIPT_DIR/docker_tests.sh"
        run_docker_tests "${FILTERED_ARGS[@]}"
        EXIT_CODE=$?
        ;;
    replit)
        # Ensure Replit environment variables are set
        export REPLIT_ENV="replit"
        export REPL_ID="${REPL_ID:-replit_test_run}"
        
        # Source the Replit test script and run tests
        source "$SCRIPT_DIR/replit_tests.sh"
        run_replit_tests "${FILTERED_ARGS[@]}"
        EXIT_CODE=$?
        ;;
esac

# Exit with the exit code from the environment-specific test script
exit $EXIT_CODE 
