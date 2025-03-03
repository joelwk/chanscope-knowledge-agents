#!/bin/bash
# Script to run validation tests inside the Docker container

set -e  # Exit on error

# Define timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/app/test_results/chanscope_tests_docker_${TIMESTAMP}.log"

# Create test results directory if it doesn't exist
mkdir -p /app/test_results
mkdir -p /app/data/stratified

echo "Starting Chanscope validation tests..."
echo "Logs will be saved to ${LOG_FILE}"

# Ensure we're in the app directory
cd /app

# Set environment variables for testing
export TEST_MODE=true
export DATA_RETENTION_DAYS=7
export ENABLE_DATA_SCHEDULER=true
export USE_MOCK_EMBEDDINGS=true
export PYTHONPATH=/app

# Parse arguments
RUN_ALL=true
RUN_DATA_INGESTION=false
RUN_EMBEDDING=false
RUN_ENDPOINTS=false
RUN_CHANSCOPE_APPROACH=false

if [ "$1" = "--all" ] || [ -z "$1" ]; then
    RUN_ALL=true
elif [ "$1" = "--data-ingestion" ]; then
    RUN_ALL=false
    RUN_DATA_INGESTION=true
elif [ "$1" = "--embedding" ]; then
    RUN_ALL=false
    RUN_EMBEDDING=true
elif [ "$1" = "--endpoints" ]; then
    RUN_ALL=false
    RUN_ENDPOINTS=true
elif [ "$1" = "--chanscope-approach" ]; then
    RUN_ALL=false
    RUN_CHANSCOPE_APPROACH=true
elif [ "$1" = "--help" ]; then
    echo "Usage: $0 [OPTION]"
    echo "Run Chanscope tests"
    echo ""
    echo "Options:"
    echo "  --all                  Run all tests (default)"
    echo "  --data-ingestion       Run only data ingestion tests"
    echo "  --embedding            Run only embedding pipeline tests"
    echo "  --endpoints            Run only API endpoint tests"
    echo "  --chanscope-approach   Run only Chanscope approach validation tests"
    echo "  --help                 Display this help and exit"
    exit 0
else
    echo "Invalid option: $1"
    echo "Use --help for usage information"
    exit 1
fi

# Results tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    echo "-----------------------------------------------------------------------------"
    echo "Running test: $test_name"
    echo "Command: $test_cmd"
    echo "-----------------------------------------------------------------------------"
    
    eval "$test_cmd"
    local status=$?
    
    if [ $status -eq 0 ]; then
        echo "✅ Test PASSED: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "❌ Test FAILED: $test_name (exit code: $status)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    return $status
}

# Run tests based on options
ALL_TESTS_PASSED=true

# Run data ingestion tests
if [ "$RUN_ALL" = true ] || [ "$RUN_DATA_INGESTION" = true ]; then
    if [ -f "tests/test_data_ingestion.py" ]; then
        run_test "Data Ingestion" "python -m tests.test_data_ingestion" || ALL_TESTS_PASSED=false
    else
        echo "⚠️ Data ingestion test file not found, skipping"
    fi
fi

# Run embedding tests
if [ "$RUN_ALL" = true ] || [ "$RUN_EMBEDDING" = true ]; then
    if [ -f "tests/test_embedding_pipeline.py" ]; then
        run_test "Embedding Pipeline" "python -m tests.test_embedding_pipeline" || ALL_TESTS_PASSED=false
    else
        echo "⚠️ Embedding pipeline test file not found, skipping"
    fi
fi

# Run endpoint tests
if [ "$RUN_ALL" = true ] || [ "$RUN_ENDPOINTS" = true ]; then
    if [ -f "tests/test_endpoints.py" ]; then
        run_test "API Endpoints" "python -m tests.test_endpoints" || ALL_TESTS_PASSED=false
    else
        echo "⚠️ API endpoints test file not found, skipping"
    fi
fi

# Run Chanscope approach validation
if [ "$RUN_ALL" = true ] || [ "$RUN_CHANSCOPE_APPROACH" = true ]; then
    run_test "Chanscope Approach Validation" "python -m scripts.validate_chanscope_approach" || ALL_TESTS_PASSED=false
fi

# Print test summary
echo "-----------------------------------------------------------------------------"
echo "Test Summary:"
echo "  Tests Run: $TESTS_RUN"
echo "  Tests Passed: $TESTS_PASSED"
echo "  Tests Failed: $TESTS_FAILED"
echo "-----------------------------------------------------------------------------"

# Check if all tests passed
if [ "$ALL_TESTS_PASSED" = true ]; then
    echo "All Chanscope tests passed successfully!"
    exit 0
else
    echo "Some Chanscope tests failed. Check the logs for details."
    exit 1
fi 