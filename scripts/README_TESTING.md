# Chanscope Testing Guide

This document provides instructions for running tests to validate the Chanscope approach implementation in different environments.

## Overview

The Chanscope approach defines a specific data processing pipeline with well-defined behaviors for different scenarios:

1. **Initial Data Load**: On application startup, data is ingested and stratified, but embedding generation is deferred.
2. **Separate Embedding Generation**: Embeddings are generated as a separate step after initial data load.
3. **Incremental Updates**: When `force_refresh=false`, existing data is used without regeneration.
4. **Forced Refresh**: When `force_refresh=true`, stratified data and embeddings are always refreshed.

## Running Tests

### Local Environment

To run tests in your local environment:

```bash
# Run all tests
bash scripts/run_tests.sh

# Run specific tests directly with pytest
python -m pytest tests/test_chanscope_approach.py -v
```

### Docker Environment

To run tests in a Docker environment:

```bash
# Run all tests in Docker
bash scripts/run_docker_tests.sh
```

### Replit Environment

To run tests in a Replit environment:

```bash
# Run tests with Replit-specific settings
bash scripts/replit/run_tests.sh
```

## Test Results

Test results are saved to the `test_results` directory with the following files:

- `chanscope_tests_TIMESTAMP.log`: Log file with detailed test output
- `chanscope_validation_ENV_TIMESTAMP.json`: JSON file with structured test results

## Test Descriptions

The test suite validates the following aspects of the Chanscope approach:

1. **Initial Data Load Test**: Validates that data is ingested and stratified, but embedding generation is skipped.
2. **Embedding Generation Test**: Validates that embeddings can be generated separately.
3. **force_refresh=false Test**: Validates that existing data is used without regeneration.
4. **force_refresh=true Test**: Validates that stratified data and embeddings are always refreshed.

## Troubleshooting

If tests fail, check the following:

1. Ensure all dependencies are installed
2. Check that data directories exist and are writable
3. Verify that environment variables are set correctly
4. Check the log files for detailed error messages

## Environment Variables

The following environment variables affect test behavior:

- `DATA_RETENTION_DAYS`: Number of days of data to retain (default: 7)
- `ENABLE_DATA_SCHEDULER`: Whether to enable the data scheduler (default: true)
- `DATA_UPDATE_INTERVAL`: Interval in seconds between data updates (default: 1800)
- `TEST_MODE`: Set to true when running tests (default: false)

# Chanscope Testing Documentation

This document describes how to run tests for the Chanscope application, focusing on Docker-based testing that follows the Chanscope approach rules.

## Docker-Based Testing

The primary script for running Docker-based tests is `scripts/run_docker_tests.sh`. This script has been enhanced to support the Chanscope approach for data management and testing.

### Basic Usage

```bash
# Run all tests with default settings
./scripts/run_docker_tests.sh

# Run only embedding tests
./scripts/run_docker_tests.sh --embedding

# Run tests with data refresh forced
./scripts/run_docker_tests.sh --force-refresh

# See all available options
./scripts/run_docker_tests.sh --help
```

### Key Features

The Docker testing script supports several features that align with the Chanscope approach:

1. **Automatic Data Checking**: By default, the script checks if the data is up-to-date according to the Chanscope approach rules and refreshes it only if needed (when data is missing, embeddings are missing, or data is older than specified retention period).

2. **Force Refresh Option**: Use `--force-refresh` to explicitly force data regeneration regardless of current status.

3. **Setup Script Integration**: The `--use-setup` option runs tests through the `setup.sh` script, which provides a more complete environment that matches production.

4. **Test Selection**: You can select specific test categories to run:
   - `--all`: Run all tests (default)
   - `--data-ingestion`: Test only the data ingestion process
   - `--embedding`: Test only the embedding generation functionality
   - `--endpoints`: Test only the API endpoints
   - `--chanscope-approach`: Test the complete Chanscope approach pipeline

### Advanced Options

```bash
# Use the setup.sh script for testing (follows Chanscope approach)
./scripts/run_docker_tests.sh --use-setup

# Disable automatic data checking (use existing data as is)
./scripts/run_docker_tests.sh --no-auto-check-data

# Combine multiple options
./scripts/run_docker_tests.sh --embedding --force-refresh --use-setup
```

## How Testing Works with the Chanscope Approach

When running with `--use-setup`, the testing process follows the Chanscope approach:

1. **Data Verification**: 
   - Checks if `complete_data.csv` exists
   - Verifies if the file is up-to-date based on DATA_RETENTION_DAYS
   - Checks if embeddings exist

2. **Data Handling**:
   - If data is missing or outdated, runs data ingestion
   - If embeddings are missing, generates new ones
   - If `--force-refresh` is used, regenerates stratified sample and embeddings

3. **Test Execution**:
   - Runs the specified tests after data preparation
   - Logs detailed results to the test_results directory

This approach ensures that tests run in an environment that closely matches production while following the data processing guidelines defined in the Chanscope approach.

## Example Use Cases

1. **CI/CD Integration**: 
   ```bash
   ./scripts/run_docker_tests.sh --all --use-setup --force-refresh
   ```
   
2. **Development Testing**:
   ```bash
   ./scripts/run_docker_tests.sh --embedding --use-setup
   ```
   
3. **Quick API Tests**:
   ```bash
   ./scripts/run_docker_tests.sh --endpoints
   ```

## Troubleshooting

If tests fail, check the detailed log file in the `test_results` directory. The logs include information about data status, test execution, and any errors encountered.

## Recommended Testing Workflow ✅

For clarity and maintainability, we recommend running tests in isolation using the dedicated testing compose file:

```bash
docker-compose -f deployment/docker-compose.test.yml build
docker-compose -f deployment/docker-compose.test.yml up
```

- Ensures isolated test data and resources.
- Optimized environment variables and resource allocation for testing.

### Integrated Testing (Advanced, Optional) ⚠️

Running tests directly in the production compose file (`docker-compose.yml`) with `RUN_TESTS_ON_STARTUP=true` is possible but should be reserved for staging or pre-production environments only due to potential risks:

- Resource contention
- Data integrity risks
- Complexity in debugging 