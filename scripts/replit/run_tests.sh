#!/bin/bash

# Script to run Chanscope tests in Replit environment
# This script is specifically designed for Replit's environment

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="test_results/chanscope_tests_replit_${TIMESTAMP}.log"

# Create test results directory if it doesn't exist
mkdir -p test_results

echo -e "${YELLOW}Starting Chanscope tests in Replit environment...${NC}"
echo -e "${YELLOW}Logs will be saved to ${LOG_FILE}${NC}"

# Ensure we're in the project root directory
cd "$(dirname "$0")/../.."

# Set environment variables for testing
export TEST_MODE=true
export DATA_RETENTION_DAYS=7
export ENABLE_DATA_SCHEDULER=true
export REPLIT_ENV=true

echo -e "${YELLOW}Running Chanscope validation tests...${NC}" | tee -a "$LOG_FILE"

# Run the validation script
python scripts/validate_chanscope_approach.py --output "test_results/chanscope_validation_replit_${TIMESTAMP}.json" 2>&1 | tee -a "$LOG_FILE"

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All Chanscope tests passed successfully!${NC}" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${RED}Some Chanscope tests failed. Check the logs for details.${NC}" | tee -a "$LOG_FILE"
    exit 1
fi 