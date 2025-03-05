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
# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key

# AWS Configuration (Required for data access)
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

## Recommended Workflow ✅

We strongly recommend the following workflow for clarity, safety, and maintainability:

### Step 1: Run Tests in Isolation (Recommended)

```bash
docker-compose -f deployment/docker-compose.test.yml build
docker-compose -f deployment/docker-compose.test.yml up
```

- Ensures isolated test data and resources.
- Optimized environment variables and resource allocation for testing.

### Step 2: Deploy to Production (Tests Disabled)

After successful testing, deploy your application without running tests at startup:

```bash
docker-compose -f deployment/docker-compose.yml build
docker-compose -f deployment/docker-compose.yml up -d
```

- Ensures stable, predictable startup behavior.
- Avoids resource contention and data integrity risks.

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

### 3. Integrated Test and Deploy Workflow (Advanced, Optional) ⚠️

Running tests directly in the production compose file (`docker-compose.yml`) with `RUN_TESTS_ON_STARTUP=true` is possible but introduces complexity and potential risks:

- **Resource Contention:** Tests and application startup processes may compete for resources, causing delays or failures.
- **Data Integrity Risks:** Tests might inadvertently modify or corrupt production data if isolation isn't perfect.
- **Complexity in Debugging:** Mixing test and production environments complicates debugging and troubleshooting.

### Recommended Usage:

- Reserve integrated testing for staging or pre-production environments only.
- Ensure robust isolation, resource allocation, and data handling if choosing this approach.
- Clearly document and communicate this decision within your team.

### Recommended Alternative:

Use the dedicated testing compose file (`docker-compose.test.yml`) for isolated, controlled testing environments.

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
- `logs`: Application logs
- `temp`: Temporary files

### Test Volumes
- `test_data`: Isolated test data
- `test_data_stratified`: Isolated stratified test data
- `test_data_shared`: Isolated shared test data
- `test_logs`: Test logs
- `test_temp`: Temporary test files

## Testing Deployment

To run tests in a Docker environment:

```bash
docker-compose -f deployment/docker-compose.test.yml build
docker-compose -f deployment/docker-compose.test.yml up
```

This will:
1. Build the Docker image
2. Start the container in test mode
3. Run the test suite
4. Output test results to the console and test_results directory

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

## Troubleshooting

### Permission Issues

If you encounter permission issues with Docker volumes:

1. Ensure the volumes are properly configured in docker-compose.yml
2. Check that the container is running as the nobody:nogroup user
3. Verify that the directories have appropriate permissions (777 for data directories)

### Data Refresh Issues

If data is not being refreshed:

1. Check the scheduler logs: `docker-compose -f deployment/docker-compose.yml exec app cat /app/data/logs/scheduler.log`
2. Verify AWS credentials are correct
3. Check S3 bucket accessibility

### Container Health Issues

If the container is not healthy:

1. Check the health status: `docker ps`
2. View container logs: `docker-compose -f deployment/docker-compose.yml logs -f`
3. Verify the API is running: `curl http://localhost/api/v1/health`

### Test Failures During Startup

If tests fail during the integrated test and deploy workflow:

1. Check the test logs: `docker-compose -f deployment/docker-compose.yml logs | grep "test"`
2. Examine test results in the test_results directory:
   ```bash
   docker-compose -f deployment/docker-compose.yml exec app ls -la /app/test_results
   docker-compose -f deployment/docker-compose.yml exec app cat /app/test_results/chanscope_tests_*.log
   ```
3. If `ABORT_ON_TEST_FAILURE=false` (default), the application will continue to run despite test failures
4. To abort deployment on test failures, set `ABORT_ON_TEST_FAILURE=true`

## Security Considerations

- The application runs as the non-root user `nobody:nogroup`
- Sensitive environment variables are passed through the `.env` file
- Data directories have appropriate permissions
- No sensitive credentials are stored in the container
- Environment variables containing API keys are marked as sensitive

## Alignment with Chanscope Approach

This Docker setup aligns with the Chanscope approach requirements:

1. **Data Processing Pipeline**:
   - Initial data ingestion from S3
   - Data stratification
   - Embedding generation

2. **Query Processing Requirements**:
   - Support for force refresh capability
   - Use of existing embeddings when available

3. **Testing Framework Support**:
   - Validation of initial data load
   - Embedding generation testing
   - Incremental processing testing
   - Force refresh behavior testing

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