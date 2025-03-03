#!/bin/bash

# Script to run Chanscope tests in Docker environment
# This script uses the test-specific Docker configuration

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
DOCKER_COMPOSE_FILE="../deployment/docker-compose.test.yml"
SHOW_LOGS="true"
CLEAN_VOLUMES="false"

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
    echo "  --clean                 Clean test volumes before running tests"
    echo "  --no-logs               Don't show logs during test execution"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --embedding --force-refresh"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            TEST_TYPE="all"
            shift
            ;;
        --data-ingestion)
            TEST_TYPE="data-ingestion"
            shift
            ;;
        --embedding)
            TEST_TYPE="embedding"
            shift
            ;;
        --endpoints)
            TEST_TYPE="endpoints"
            shift
            ;;
        --chanscope-approach)
            TEST_TYPE="chanscope-approach"
            shift
            ;;
        --force-refresh)
            FORCE_REFRESH="true"
            shift
            ;;
        --auto-check-data)
            AUTO_CHECK_DATA="true"
            shift
            ;;
        --no-auto-check-data)
            AUTO_CHECK_DATA="false"
            shift
            ;;
        --clean)
            CLEAN_VOLUMES="true"
            shift
            ;;
        --no-logs)
            SHOW_LOGS="false"
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

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Ensure test_results directory exists
mkdir -p "$PROJECT_ROOT/test_results"

# Clean test volumes if requested
if [ "$CLEAN_VOLUMES" = "true" ]; then
    echo "Cleaning test volumes..."
    docker-compose -f deployment/docker-compose.test.yml down -v
    echo "Test volumes cleaned."
fi

# Build the test image
echo "Building test Docker image..."
docker-compose -f deployment/docker-compose.test.yml build

# Run the tests with appropriate environment variables
echo "Running tests with TEST_TYPE=$TEST_TYPE, FORCE_REFRESH=$FORCE_REFRESH, AUTO_CHECK_DATA=$AUTO_CHECK_DATA"

# Create a timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/test_results/chanscope_tests_docker_$TIMESTAMP.log"

# Run the tests
if [ "$SHOW_LOGS" = "true" ]; then
    # Show logs in real-time
    docker-compose -f deployment/docker-compose.test.yml run --rm \
        -e TEST_MODE=true \
        -e TEST_TYPE="$TEST_TYPE" \
        -e FORCE_DATA_REFRESH="$FORCE_REFRESH" \
        -e AUTO_CHECK_DATA="$AUTO_CHECK_DATA" \
        chanscope-test | tee "$LOG_FILE"
else
    # Run silently and save logs to file
    echo "Running tests silently, logs will be saved to $LOG_FILE"
    docker-compose -f deployment/docker-compose.test.yml run --rm \
        -e TEST_MODE=true \
        -e TEST_TYPE="$TEST_TYPE" \
        -e FORCE_DATA_REFRESH="$FORCE_REFRESH" \
        -e AUTO_CHECK_DATA="$AUTO_CHECK_DATA" \
        chanscope-test > "$LOG_FILE" 2>&1
fi

# Check exit code
EXIT_CODE=$?

# Display test results summary
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed successfully!"
else
    echo "❌ Some tests failed with exit code: $EXIT_CODE"
    echo "Check the logs for details: $LOG_FILE"
fi

# Look for test result JSON files and display a summary if they exist
RESULT_FILES=$(find "$PROJECT_ROOT/test_results" -name "chanscope_validation_docker_*.json" -type f -newer "$LOG_FILE")
if [ -n "$RESULT_FILES" ]; then
    echo "Test result files:"
    for file in $RESULT_FILES; do
        echo "  - $file"
        # If jq is available, use it to display a summary
        if command -v jq &> /dev/null; then
            echo "Summary:"
            jq -r '.summary // "No summary available"' "$file" 2>/dev/null || echo "  Could not parse JSON file"
            echo "Status: $(jq -r '.status // "Unknown"' "$file" 2>/dev/null)"
            echo ""
        fi
    done
fi

# Clean up if needed
if [ "$CLEAN_VOLUMES" = "true" ]; then
    echo "Cleaning up test environment..."
    docker-compose -f deployment/docker-compose.test.yml down
fi

exit $EXIT_CODE 