#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure required directories exist
mkdir -p "$WORKSPACE_ROOT/data"
mkdir -p "$WORKSPACE_ROOT/logs"

# Set up Python path
export PYTHONPATH="$WORKSPACE_ROOT:$PYTHONPATH"

echo -e "${YELLOW}Running process lock tests...${NC}"

# Process command line arguments
TEST_TYPE="all"
if [ $# -gt 0 ]; then
    TEST_TYPE="$1"
fi

# Identify the correct Python command (python3 or python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Set environment for test
if [ -z "$ENVIRONMENT" ]; then
    # Try to detect environment
    if [ -f "/.dockerenv" ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
        echo -e "${YELLOW}Detected Docker environment${NC}"
        export DOCKER_ENV=true
    elif [ -n "$REPL_ID" ] || [ -n "$REPL_SLUG" ] || [ -n "$REPLIT_ENV" ]; then
        echo -e "${YELLOW}Detected Replit environment${NC}"
        export REPLIT_ENV=replit
    else
        echo -e "${YELLOW}Assuming local environment${NC}"
    fi
fi

# Run the appropriate test
case $TEST_TYPE in
    acquisition|acquire)
        echo -e "${YELLOW}Running lock acquisition test...${NC}"
        $PYTHON_CMD "$WORKSPACE_ROOT/scripts/test_process_lock.py" --test-acquire
        ;;
    contention)
        echo -e "${YELLOW}Running lock contention test...${NC}"
        $PYTHON_CMD "$WORKSPACE_ROOT/scripts/test_process_lock.py" --test-contention
        ;;
    marker)
        echo -e "${YELLOW}Running initialization marker test...${NC}"
        $PYTHON_CMD "$WORKSPACE_ROOT/scripts/test_process_lock.py" --test-marker
        ;;
    all)
        echo -e "${YELLOW}Running all process lock tests...${NC}"
        $PYTHON_CMD "$WORKSPACE_ROOT/scripts/test_process_lock.py" --all
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo -e "${YELLOW}Available options: acquisition, contention, marker, all${NC}"
        exit 1
        ;;
esac

# Check if the test was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Process lock tests completed successfully!${NC}"
else
    echo -e "${RED}Process lock tests failed!${NC}"
    exit 1
fi 