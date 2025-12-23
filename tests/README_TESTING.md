# Chanscope Retrieval Testing Framework

This document provides an overview of the Chanscope Retrieval testing framework, including how to run tests in different environments and the structure of the test scripts.

## Test Structure

The Chanscope Retrieval testing framework is designed to work across multiple environments:

- **Local**: For running tests on your local machine
- **Docker**: For running tests in a Docker container
- **Replit**: For running tests in a Replit environment

The test scripts are organized as follows:

- `run_tests.sh`: Unified test runner with environment detection (local, Docker, Replit)
- `test_and_deploy.sh`: Runs tests and then deploys if tests pass

## Test Categories

The Chanscope Retrieval testing framework includes the following test categories:
- Data Ingestion Tests
- Embedding Tests
- API Endpoint Tests
- Chanscope Retrieval Approach Tests

## Running Tests

### Basic Usage

To run all tests in the current environment:

```bash
scripts/run_tests.sh
```

The script will automatically detect whether you're running in a local, Docker, or Replit environment.

### Specifying the Environment

You can explicitly specify the environment:

```bash
scripts/run_tests.sh --env=local
scripts/run_tests.sh --env=docker
scripts/run_tests.sh --env=replit
```

### Running Specific Test Categories

You can run specific test categories:

```bash
# Run only data ingestion tests
scripts/run_tests.sh --data-ingestion

# Run only embedding tests
scripts/run_tests.sh --embedding

# Run only API endpoint tests
scripts/run_tests.sh --endpoints

# Run only Chanscope approach validation tests
scripts/run_tests.sh --chanscope-approach
```

### Additional Options

```bash
# Force data refresh before tests
scripts/run_tests.sh --force-refresh

# Disable automatic data checking
scripts/run_tests.sh --no-auto-check-data

# Use real data instead of mock data (local/Replit only)
scripts/run_tests.sh --no-mock-data

# Use real embeddings instead of mock embeddings (local/Replit only)
scripts/run_tests.sh --no-mock-embeddings

# Clean test volumes before running tests (Docker only)
scripts/run_tests.sh --env=docker --clean

# Don't show logs during test execution
scripts/run_tests.sh --no-logs
```

## Test and Deploy Workflow

The `test_and_deploy.sh` script provides a workflow for running tests and then deploying if tests pass:

```bash
# Run tests and deploy as separate steps (default)
scripts/test_and_deploy.sh

# Specify the environment for testing
scripts/test_and_deploy.sh --env=docker

# Use integrated test and deploy workflow
scripts/test_and_deploy.sh --integrated
```

## Configuration

Test configuration is centralized in the `.env.test` file, which includes:

- Shared configuration for all environments
- Environment-specific configuration for local, Docker, and Replit environments

## Test Results

Test results are saved in the `test_results` directory, including:

- Log files with detailed test output
- JSON result files with test summary information
- JUnit XML files for CI/CD integration

## Docker-Based Testing

The primary script for running Docker-based tests is `scripts/run_tests.sh --env=docker`. This script has been enhanced to support the Chanscope approach for data management and testing.

### Basic Usage

```bash
# Run all tests with default settings
./scripts/run_tests.sh --env=docker

# Run only embedding tests
./scripts/run_tests.sh --env=docker --embedding

# Run tests with data refresh forced
./scripts/run_tests.sh --env=docker --force-refresh

# See all available options
./scripts/run_tests.sh --help
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
./scripts/run_tests.sh --env=docker --use-setup

# Disable automatic data checking (use existing data as is)
./scripts/run_tests.sh --env=docker --no-auto-check-data

# Combine multiple options
./scripts/run_tests.sh --env=docker --embedding --force-refresh --use-setup
```

## Troubleshooting

### Common Issues

- **Permission Issues**: Ensure you have write permissions to the directories used by the tests.
- **Docker Volume Issues**: Use the `--clean` option to clean test volumes before running tests.
- **Python Not Found**: Ensure Python is installed and available in your PATH.
- **Missing Dependencies**: Ensure all required dependencies are installed.

### Environment-Specific Issues

#### Local Environment

- Ensure Python and required packages are installed.
- Check that you have write permissions to the test directories.

#### Docker Environment

- Ensure Docker and Docker Compose are installed.
- Check that the Docker daemon is running.
- Verify that the Docker Compose files are correctly configured.

#### Replit Environment

- Ensure the Replit environment has the necessary permissions.
- Check that the `REPL_HOME` environment variable is set.
- Verify that Python is available in the Replit environment.

## Complete Test Workflow Example

Here's a complete workflow for testing the knowledge agent:

```bash
# 1. Check health
curl -X GET "${BASE_URL}${API_PATH}/health"

# 2. Trigger data stratification
curl -X POST "${BASE_URL}/api/stratify"

# 3. Trigger embedding generation
curl -X POST "${BASE_URL}/api/trigger_embedding_generation"

# 4. Check embedding status
curl -X GET "${BASE_URL}/api/embedding_status"

# 5. Submit a background query
RESPONSE=$(curl -X POST "${BASE_URL}${API_PATH}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Latest developments in AI and machine learning",
    "use_background": true
  }')

# 6. Extract task_id from response
TASK_ID=$(echo $RESPONSE | jq -r '.task_id')
echo "Task ID: $TASK_ID"

# 7. Check query status
curl -X GET "${BASE_URL}${API_PATH}/batch_status/$TASK_ID"
```

### Task Status Tracking

The Knowledge Agent now includes an enhanced task status tracking system that provides detailed information about the status of background tasks:

- **Task Status Persistence**: Task status information is now persisted to disk in a `batch_history.json` file, allowing status retrieval even after the task has been removed from memory.
- **Improved Status Responses**: The `/batch_status/{task_id}` endpoint now provides more detailed status information, including:
  - Processing status (queued, processing, completed, failed, expired)
  - Detailed error messages when tasks fail
  - Timestamps for task creation and completion
  - Position in queue and estimated processing time for queued tasks

Example response for a completed task:
```json
{
  "status": "completed",
  "result": {
    "summary": "Analysis of investment opportunities in renewable energy...",
    "sources": [...]
  }
}
```

Example response for an expired task:
```json
{
  "status": "expired",
  "message": "Task task_1234567890_abcdef was completed but results have expired. Results are only kept for 10 minutes.",
  "completed_at": "2023-06-01T12:34:56.789Z",
  "task_id": "task_1234567890_abcdef"
}
```

### Periodic Cleanup

The system now includes a periodic cleanup process that:
- Removes old task results to prevent memory leaks
- Updates the batch history file with final status information
- Maintains a record of completed tasks for future reference

## Future Improvements

As part of our ongoing consolidation efforts:

1. We plan to create a unified test runner that works across all environments (Docker, Replit, local).
2. Update CI/CD and deployment scripts to use the consolidated scripts.
3. Further streamline the testing workflow to reduce duplication.
4. Enhance the task management system with additional metrics and monitoring capabilities. 
