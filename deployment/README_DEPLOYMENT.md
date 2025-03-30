# Knowledge Agent Deployment

This directory contains the Docker configuration files for deploying the Knowledge Agent application in both production and testing environments.

## Prerequisites

- Docker and Docker Compose installed
- AWS credentials for S3 access (required for data retrieval)
- OpenAI API key (required for embeddings and query processing)
- Optional: Grok and Venice API keys for multi-provider capabilities

## Environment Setup

1. Create an environment file by copying the template:

```bash
cp .env.template .env
```

2. Edit the `.env` file to set the required environment variables:

```
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_aws_region
S3_BUCKET=your_s3_bucket

# Optional Additional Provider Keys
GROK_API_KEY=your_grok_api_key
VENICE_API_KEY=your_venice_api_key

# Application Configuration
LOG_LEVEL=info
API_PORT=80
API_WORKERS=4
```

3. For testing, create a test environment file:

```bash
cp .env.template .env.test
```

4. Edit the `.env.test` file with appropriate test values.

## Docker Configuration Files

The deployment directory contains several important files:

- `docker-compose.yml`: The main Docker Compose file for production deployment
- `docker-compose.test.yml`: Docker Compose file configured for testing
- `Dockerfile`: Defines the container image
- `setup.sh`: Initialization script that runs when the container starts

## Recommended Deployment Workflow
1. Run tests in isolation using docker-compose.test.yml
2. Deploy to production using docker-compose.yml
3. Monitor deployment using the health check endpoints

## Deployment Options

### 1. Production Deployment

To deploy the application in production mode without running tests:

```bash
docker-compose -f deployment/docker-compose.yml build
docker-compose -f deployment/docker-compose.yml up -d
```

This will:
1. Build the Docker image
2. Start the container in detached mode
3. Initialize the data directories
4. Fetch data from S3 if needed
5. Start the data scheduler
6. Launch the API service

### 2. Testing Before Deployment (Separate Steps)

To run tests first and then deploy to production:

```bash
# Step 1: Run tests
docker-compose -f deployment/docker-compose.test.yml build
docker-compose -f deployment/docker-compose.test.yml up

# Step 2: If tests pass, deploy to production
docker-compose -f deployment/docker-compose.yml build
docker-compose -f deployment/docker-compose.yml up -d
```

## Docker Environment Variables

Key environment variables used in the Docker setup:

### Production Variables
- `OPENAI_API_KEY`: Required for OpenAI API access
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`: Required for S3 access
- `S3_BUCKET`: The S3 bucket containing 4chan data
- `API_PORT`: Port on which the API will be exposed (default: 80)
- `API_WORKERS`: Number of Gunicorn workers (default: 4)
- `LOG_LEVEL`: Logging level (default: info)
- `DATA_REFRESH_INTERVAL`: How often to refresh data (in hours, default: 1)
- `DATA_RETENTION_DAYS`: How many days of data to retain (default: 30)

### Testing Variables
- `TEST_MODE`: Set to true for test mode
- `RUN_TESTS_ON_STARTUP`: Set to true to run tests on startup
- `TEST_TYPE`: Type of tests to run (default: all)
- `FORCE_REFRESH`: Force data refresh before tests (default: false)
- `AUTO_CHECK_DATA`: Automatically check data before tests (default: true)
- `ABORT_ON_TEST_FAILURE`: Abort deployment on test failure (default: false)

## Data Management

The Knowledge Agent uses several Docker volumes to manage data:

### Production Volumes
- `data`: Main data directory
- `data_stratified`: Stratified data samples
- `data_shared`: Shared data for embeddings and tokens
- `logs`: Application logs (includes batch_history.json for task tracking)
- `temp`: Temporary files

### Test Volumes
- `test_data`: Isolated test data
- `test_data_stratified`: Isolated stratified test data
- `test_data_shared`: Isolated shared test data
- `test_logs`: Test logs (includes batch_history.json for test task tracking)
- `test_temp`: Temporary test files

### Task Management

The Knowledge Agent includes a robust task management system that:

1. **Tracks Background Tasks**: Maintains status information for all background tasks
2. **Persists Task History**: Stores task history in a `batch_history.json` file in the logs directory
3. **Provides Detailed Status**: Offers detailed status information through the `/api/v1/batch_status/{task_id}` endpoint
4. **Performs Automatic Cleanup**: Periodically removes old task results to prevent memory leaks
5. **Preserves Task History**: Maintains a record of completed tasks even after results are removed from memory

This system ensures that users can track the status of their queries even if they were submitted hours ago, while preventing memory issues from accumulating task results.

## Monitoring and Maintenance

### Checking Container Status

```bash
docker-compose -f deployment/docker-compose.yml ps
```

### Viewing Logs

```bash
docker-compose -f deployment/docker-compose.yml logs -f
```

### Restarting the Application

```bash
docker-compose -f deployment/docker-compose.yml restart
```

### Stopping the Application

```bash
docker-compose -f deployment/docker-compose.yml down
```

## Automated Deployment

The project includes scripts for automated deployment:

- `scripts/test_and_deploy.sh`: Runs tests and deploys if tests pass
- `deployment/setup.sh`: Sets up the container environment on startup

## Replit Deployment

For Replit deployment, the project includes specialized configuration:

- Environment variables are set through Replit Secrets
- Path handling is adapted for Replit's filesystem
- Memory management is optimized for Replit's constraints
- Startup is handled through Replit's run button

The Knowledge Agent should work seamlessly on Replit with minimal configuration changes.