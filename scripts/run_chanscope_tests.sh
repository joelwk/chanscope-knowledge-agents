#!/bin/bash
set -e

# Chanscope Test Runner
# This script runs all tests to validate the Chanscope approach implementation
# in both Docker and Replit environments.

# Get the application root directory
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${APP_ROOT}/test_results"
mkdir -p "${RESULTS_DIR}"

# Set timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULTS_DIR}/chanscope_tests_${TIMESTAMP}.log"

# Detect environment
if [ -f /.dockerenv ]; then
    ENV_TYPE="docker"
elif [ -n "$REPL_ID" ]; then
    ENV_TYPE="replit"
else
    ENV_TYPE="local"
fi

echo "Running Chanscope Tests in ${ENV_TYPE} environment"
echo "================================================="
echo "Test results will be saved to: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"
echo

# Function to log messages
log() {
    local message="$1"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" | tee -a "${LOG_FILE}"
}

# Function to run a test and log results
run_test() {
    local test_name="$1"
    local command="$2"
    
    log "Running test: ${test_name}"
    log "Command: ${command}"
    
    # Run the command and capture output
    eval "${command}" 2>&1 | tee -a "${LOG_FILE}"
    local status=${PIPESTATUS[0]}
    
    if [ $status -eq 0 ]; then
        log "✅ Test passed: ${test_name}"
    else
        log "❌ Test failed: ${test_name} (exit code: ${status})"
    fi
    
    echo
    return $status
}

# Set environment variables for testing
export DATA_RETENTION_DAYS=7
export ENABLE_DATA_SCHEDULER=true
export DATA_UPDATE_INTERVAL=1800

# Log environment information
log "Test environment:"
log "- Environment type: ${ENV_TYPE}"
log "- DATA_RETENTION_DAYS: ${DATA_RETENTION_DAYS}"
log "- ENABLE_DATA_SCHEDULER: ${ENABLE_DATA_SCHEDULER}"
log "- DATA_UPDATE_INTERVAL: ${DATA_UPDATE_INTERVAL}"
log "- Python version: $(python --version 2>&1)"
log "- Operating system: $(uname -a)"

# Run the comprehensive validation script
log "Running comprehensive Chanscope validation"
VALIDATION_OUTPUT="${RESULTS_DIR}/chanscope_validation_${ENV_TYPE}_${TIMESTAMP}.json"
run_test "Comprehensive Validation" "cd ${APP_ROOT} && poetry run python scripts/validate_chanscope_approach.py --output ${VALIDATION_OUTPUT}"
VALIDATION_STATUS=$?

# Run the individual test script
log "Running individual Chanscope tests"
run_test "Individual Tests" "cd ${APP_ROOT} && bash scripts/test_chanscope_implementation.sh"
INDIVIDUAL_STATUS=$?

# Test scheduled update
log "Testing scheduled update"
run_test "Scheduled Update" "cd ${APP_ROOT} && poetry run python scripts/scheduled_update.py --run_once"
SCHEDULED_STATUS=$?

# Print summary
echo
echo "Chanscope Test Summary"
echo "======================"
echo "Environment: ${ENV_TYPE}"
echo "Timestamp: $(date)"
echo

echo "Test Results:"
if [ $VALIDATION_STATUS -eq 0 ]; then
    echo "- Comprehensive Validation: ✅ PASS"
else
    echo "- Comprehensive Validation: ❌ FAIL"
fi

if [ $INDIVIDUAL_STATUS -eq 0 ]; then
    echo "- Individual Tests: ✅ PASS"
else
    echo "- Individual Tests: ❌ FAIL"
fi

if [ $SCHEDULED_STATUS -eq 0 ]; then
    echo "- Scheduled Update: ✅ PASS"
else
    echo "- Scheduled Update: ❌ FAIL"
fi

echo
echo "Detailed results saved to: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Validation results: ${VALIDATION_OUTPUT}"

# Set exit status based on all tests
if [ $VALIDATION_STATUS -eq 0 ] && [ $INDIVIDUAL_STATUS -eq 0 ] && [ $SCHEDULED_STATUS -eq 0 ]; then
    log "All tests passed!"
    exit 0
else
    log "Some tests failed. See log for details."
    exit 1
fi 