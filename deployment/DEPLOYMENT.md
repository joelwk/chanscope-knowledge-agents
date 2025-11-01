# Knowledge Agent Deployment Guide

This is the **single source of truth** for deploying the Knowledge Agent application. It covers both Docker and Replit deployment environments with comprehensive setup, configuration, and operational guidance.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Detection](#environment-detection)
- [Docker Deployment](#docker-deployment)
- [Replit Deployment](#replit-deployment)
- [Environment Configuration](#environment-configuration)
- [Data Management](#data-management)
- [API Reference](#api-reference)
- [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
- [Refresh Dashboard](#refresh-dashboard)

## Prerequisites

### Required

- **Docker and Docker Compose** (for Docker deployment)
- **OpenAI API Key** (required for embeddings and query processing)
- **AWS Credentials** (for S3 data access):
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`
  - `S3_BUCKET`

### Optional

- **Grok API Key** (for multi-provider capabilities)
- **Venice API Key** (for multi-provider capabilities)
- **PostgreSQL Database** (required for Replit deployment and NL→SQL queries)

## Environment Detection

The system automatically detects the deployment environment and configures storage accordingly:

### Docker Environment

**Detection**: Automatically detected when:
- `/.dockerenv` file exists
- `ENVIRONMENT=docker` environment variable is set
- `DOCKER_ENV=true` environment variable is set

**Storage Backend**: File-based storage
- Complete data: `/app/data/complete_data.csv`
- Stratified samples: `/app/data/stratified/stratified_sample.csv`
- Embeddings: `/app/data/stratified/embeddings.npz`
- Thread ID mapping: `/app/data/stratified/thread_id_map.json`
- Process locks: File-based locks

### Replit Environment

**Detection**: Automatically detected when:
- `REPL_ID`, `REPL_SLUG`, or `REPL_OWNER` environment variables are present
- `/home/runner` directory exists
- `REPLIT_ENV=replit` environment variable is set

**Storage Backend**: Database and cloud storage
- Complete data: PostgreSQL database (`complete_data` table)
- Stratified samples: Replit Key-Value store
- Embeddings: Replit Object Storage (compressed .npz format)
- Process locks: Object Storage for persistence across restarts

**Note**: Natural Language to SQL queries (`/api/v1/nl_query`) are **only available** in Replit environments where PostgreSQL is configured.

## Docker Deployment

### Quick Start

```bash
# Build and start production deployment
docker-compose -f deployment/docker-compose.yml build --no-cache
docker-compose -f deployment/docker-compose.yml up -d

# Verify environment detection (should print: docker)
docker exec $(docker ps -q) python -c "from config.env_loader import detect_environment; print(detect_environment())"
```

### Production Deployment

1. **Create environment file**:
```bash
cp .env.template .env
```

2. **Configure environment variables** in `.env`:
```bash
# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key

# AWS Configuration (Required)
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
AUTO_REFRESH_MANAGER=true
DATA_REFRESH_INTERVAL=3600
```

3. **Deploy**:
```bash
docker-compose -f deployment/docker-compose.yml build --no-cache
docker-compose -f deployment/docker-compose.yml up -d
```

4. **Verify deployment**:
```bash
# Check container status
docker-compose -f deployment/docker-compose.yml ps

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Test health endpoint
curl http://localhost/api/v1/health
```

### Testing Before Deployment

Run tests in isolation before deploying to production:

```bash
# Step 1: Run tests
docker-compose -f deployment/docker-compose.test.yml build
docker-compose -f deployment/docker-compose.test.yml up

# Step 2: If tests pass, deploy to production
docker-compose -f deployment/docker-compose.yml build
docker-compose -f deployment/docker-compose.yml up -d
```

### Docker Volumes

The Docker deployment uses the following volumes:

- `data`: Main data directory
- `data_stratified`: Stratified data samples
- `data_shared`: Shared data for embeddings and tokens
- `logs`: Application logs (includes `batch_history.json` for task tracking)
- `temp`: Temporary files

### Docker Environment Variables

The `docker-compose.yml` file explicitly sets environment variables to ensure proper detection:

```yaml
environment:
  - ENVIRONMENT=docker
  - DOCKER_ENV=true
  # Explicitly unset REPLIT_ENV to prevent conflicts
  - REPLIT_ENV=
  - AUTO_REFRESH_MANAGER=${AUTO_REFRESH_MANAGER:-true}
  - DATA_REFRESH_INTERVAL=${DATA_REFRESH_INTERVAL:-3600}
```

## Replit Deployment

### Pre-Deployment Setup

1. **Run pre-deployment verification**:
```bash
bash scripts/replit_setup.sh
```

This script validates:
- `requirements.txt` format
- FastAPI app structure
- Port configuration
- Creates default `.env` with safe settings

2. **Configure Replit Secrets** (Deployment → Secrets):

```bash
# Database (Required for production)
DATABASE_URL=your_postgresql_url
# OR individual parameters:
PGHOST=your_host
PGUSER=your_user
PGPASSWORD=your_password

# AWS S3 (Optional, for external data)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=your_region
S3_BUCKET=your_bucket

# OpenAI (Required)
OPENAI_API_KEY=your_key

# Data Processing Control (Recommended settings)
AUTO_CHECK_DATA=false
AUTO_PROCESS_DATA_ON_INIT=false
FORCE_DATA_REFRESH=false
SKIP_EMBEDDINGS=false
ENABLE_DATA_SCHEDULER=false
AUTO_REFRESH_MANAGER=false
```

> Replit automatically merges these secrets with the repository `.env`. Secrets (and any values you inject via the Replit UI) win, while the `.env` file provides defaults for anything that is missing.

### Deployment Steps

1. **Click Deploy** in Replit
2. **Select Autoscale** deployment
3. **Choose appropriate machine power**
4. **Monitor deployment logs**

### Startup Sequence

The Replit deployment uses an optimized startup sequence:

1. **Server starts immediately** and binds to port 80
2. **Health checks pass** (within 10 seconds)
3. **Background initialization** runs after server stabilization:
   - Dependencies install (if needed)
   - Database schema initialization
   - Service verification (AWS, Replit KV)

### Post-Deployment Data Initialization

After successful deployment, trigger data processing:

```bash
# Option 1: Use trigger endpoint
curl https://your-app.replit.app/trigger-data-processing

# Option 2: Use initialization endpoint
curl https://your-app.replit.app/api/v1/force-initialization
```

### Health Check Endpoints

Replit deployments respond immediately to these endpoints:

- **`/`** - Ultra-fast health check (middleware-intercepted)
- **`/healthz`** - Standard health check with timestamp
- **`/api/v1/health`** - Extended health check with environment details

### Replit-Specific Troubleshooting

#### Issue: "Port not opening in time"
**Solution**: The optimized startup sequence should fix this. If it persists:
1. Check that `python -m uvicorn api.app:app --host 0.0.0.0 --port 80` works locally
2. Verify no dependencies are missing in `requirements.txt`

#### Issue: "Health check failing"
**Solution**:
1. Test the root endpoint: `curl https://your-app.replit.app/`
2. Check if middleware is working properly
3. Verify no blocking operations in the root endpoint

#### Issue: "Database connection errors"
**Solution**:
1. Verify `DATABASE_URL` is set in deployment secrets
2. Check PostgreSQL database is accessible from Replit
3. Use `/api/v1/initialization-status` to see specific errors

## Environment Configuration

### Common Environment Variables

#### Core Configuration

```bash
# API Configuration
API_PORT=80                      # Port for API server
API_WORKERS=4                    # Number of Gunicorn workers (production)
LOG_LEVEL=info                   # Logging level (debug, info, warning, error)

# Environment Detection
ENVIRONMENT=docker               # docker or replit (auto-detected if not set)
DOCKER_ENV=true                  # Explicit Docker flag
REPLIT_ENV=                      # Explicitly unset in Docker
```

#### Data Processing Control

```bash
# Startup Behavior
AUTO_CHECK_DATA=false            # Don't auto-check data on startup
AUTO_PROCESS_DATA_ON_INIT=false # Don't auto-process data on init

# Data Refresh
FORCE_DATA_REFRESH=false        # Force refresh all data stages
SKIP_EMBEDDINGS=false           # Skip embedding generation
DATA_RETENTION_DAYS=14          # Days of data to retain
DATA_UPDATE_INTERVAL=86400      # Update interval in seconds (default: daily)
DATA_REFRESH_INTERVAL=3600      # Refresh interval for auto-refresh manager

# Processing Configuration
EMBEDDING_BATCH_SIZE=25         # Number of items per embedding batch
PROCESSING_CHUNK_SIZE=10000    # Chunk size for data processing
MAX_WORKERS=4                   # Maximum number of worker processes
```

#### Auto-Refresh Manager

```bash
# Enable/Disable Auto-Refresh
AUTO_REFRESH_MANAGER=true       # Enable automated refresh manager (default: true)
DATA_REFRESH_INTERVAL=3600      # Refresh interval in seconds (default: 3600)

# Refresh Dashboard Security
REFRESH_CONTROL_TOKEN=          # Shared secret for refresh dashboard control
```

#### Performance Tuning

```bash
FASTAPI_DEBUG=false             # Disable debug mode for faster startup
FASTAPI_ENV=production          # Production environment setting
CORS_ORIGINS=*                  # CORS origins (use specific domains in production)
```

## Data Management

### Data Processing Pipeline

The Knowledge Agent uses a three-stage data processing pipeline:

1. **Complete Data Stage**
   - Ingests data from S3 into primary storage
   - Uses PostgreSQL in Replit or CSV files in Docker
   - Configurable retention period via `DATA_RETENTION_DAYS`
   - Automatic incremental updates

2. **Stratified Sample Stage**
   - Creates representative samples from complete data
   - Uses Key-Value store in Replit or CSV files in Docker
   - Configurable sample size and stratification criteria
   - Can be regenerated independently with `--regenerate --stratified-only`

3. **Embedding Stage**
   - Generates embeddings for stratified samples
   - Uses Object Storage in Replit or NPZ files in Docker
   - Optimized batch processing for memory efficiency
   - Can be regenerated independently with `--regenerate --embeddings-only`

### Data Processing Commands

#### Initial Setup

```bash
# Process all stages (ingestion, stratification, embeddings)
python scripts/process_data.py

# Check current data status (non-invasive)
python scripts/process_data.py --check

# Force refresh all data
python scripts/process_data.py --force-refresh
```

#### Regeneration Commands

```bash
# Regenerate stratified sample only
python scripts/process_data.py --regenerate --stratified-only

# Regenerate embeddings only
python scripts/process_data.py --regenerate --embeddings-only

# Bypass process locks (use with caution)
python scripts/process_data.py --ignore-lock
```

#### Scheduled Updates

```bash
# Continuous scheduled updates (refreshes data hourly)
python scripts/scheduled_update.py refresh --continuous --interval=3600

# Continuous updates with forced stratification regeneration
python scripts/scheduled_update.py refresh --continuous --force-refresh --interval=3600

# One-time refresh
python scripts/scheduled_update.py refresh

# Check status
python scripts/scheduled_update.py status
```

### Understanding Data Refresh Behavior

#### Standard Refresh (`scheduled_update.py refresh`)
- Always processes all data files and updates the database with new records
- Reuses existing stratified samples even if they are outdated
- Reuses existing embeddings even if they are outdated
- Will only generate new stratified samples or embeddings if they don't exist

#### Force Refresh (`scheduled_update.py refresh --force-refresh`)
- Always processes all data files and updates the database with new records
- Always regenerates the stratified sample from the latest complete data
- Always regenerates embeddings from the new stratified sample
- Ensures all three stages reflect the current data state

#### Continuous Updates (`scheduled_update.py refresh --continuous --interval=3600`)
- Runs the refresh process repeatedly at the specified interval (in seconds)
- Without `--force-refresh`, will continue to use existing stratified samples
- With `--force-refresh`, will regenerate stratified samples and embeddings each cycle

**Recommended for production**: Use continuous mode with force refresh to ensure data stays current:
```bash
python scripts/scheduled_update.py refresh --continuous --force-refresh --interval=3600
```

### API Data Management Endpoints

```bash
# Trigger data stratification
curl -X POST "http://localhost/api/v1/data/stratify"

# Trigger embedding generation
curl -X POST "http://localhost/api/v1/trigger_embedding_generation"

# Check embedding generation status
curl -X GET "http://localhost/api/v1/embedding_status"

# Check initialization status
curl -X GET "http://localhost/api/v1/initialization-status"
```

## API Reference

### Base URLs

- **Local Development**: `http://localhost/api`
- **Docker Deployment**: `http://localhost/api`
- **Replit Deployment**: `https://your-app.replit.app/api`

All API version 1 endpoints are available under `/api/v1`.

### Health Check Endpoints

#### Basic Health Check
```bash
curl -X GET "http://localhost/api/v1/health"
```

**Response**:
```json
{
  "status": "ok",
  "timestamp": "2023-06-01T12:34:56.789Z",
  "environment": "docker"
}
```

#### Comprehensive Health Check
```bash
curl -X GET "http://localhost/api/v1/health/connections"
curl -X GET "http://localhost/api/v1/health/s3"
curl -X GET "http://localhost/api/v1/health/provider/openai"
curl -X GET "http://localhost/api/v1/health/all"
curl -X GET "http://localhost/api/v1/health/cache"
curl -X GET "http://localhost/api/v1/health/embeddings"
```

### Query Processing

#### Basic Query (Synchronous)

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "force_refresh": false,
    "skip_embeddings": false,
    "use_background": false
  }'
```

**Response**:
```json
{
  "status": "completed",
  "task_id": "query_1654321098_abcd",
  "chunks": [
    {
      "text": "Solar energy investments are projected to grow by 25%...",
      "score": 0.92,
      "metadata": {
        "timestamp": "2023-05-15T08:45:12Z",
        "source": "4chan",
        "thread_id": "12345678"
      }
    }
  ],
  "summary": "Recent discussions across social media platforms indicate...",
  "metadata": {
    "processing_time_ms": 1234.56,
    "num_relevant_strings": 5
  }
}
```

#### Background Query Processing

```bash
# Submit query for background processing
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "use_background": true
  }'

# Response: {"status": "processing", "task_id": "query_1654321098_abcd"}

# Check status
curl -X GET "http://localhost/api/v1/batch_status/query_1654321098_abcd"
```

#### Query with Custom Task ID

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "task_id": "my_custom_task_123",
    "use_background": true
  }'
```

#### Source-Specific Queries

```bash
# Filter to specific board/source
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "select_board": "biz"
  }'
```

#### Date-Filtered Queries

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "filter_date": "2023-05-15"
  }'
```

#### Process Recent Query

Automatically processes queries using recent data from the last 6-12 hours:

```bash
curl -X GET "http://localhost/api/v1/process_recent_query?select_board=biz"
```

### Natural Language Database Queries (Replit Only)

**Note**: This endpoint requires PostgreSQL and is only available in Replit environments.

#### Basic NL Query

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts about Bitcoin from last week",
    "limit": 100
  }'
```

#### Complex Time-Based Queries

```bash
# Posts from last 3 hours
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts about Bitcoin from the last 3 hours containing mentions of ETFs",
    "limit": 50
  }'
```

#### Multi-Filter Queries

```bash
# Complex filter query
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find threads from biz board by author john about crypto regulations from last week",
    "limit": 100
  }'
```

#### Content Analysis Queries

```bash
# What are people saying about a topic?
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are people saying about AI regulation this month?",
    "limit": 200
  }'
```

#### Supported Time Filters

- Last hour: `"last hour"`, `"past hour"`, `"previous hour"`
- Last X hours: `"last 5 hours"`, `"past 12 hours"`
- Today: `"today"`, `"this day"`
- Yesterday: `"yesterday"`, `"previous day"`
- Last X days: `"last 7 days"`, `"past 30 days"`
- This week: `"this week"`, `"current week"`
- Last week: `"last week"`, `"previous week"`
- This month: `"this month"`, `"current month"`
- Last month: `"last month"`, `"previous month"`

#### Content Filters

- `"containing [term]"`, `"with [term]"`, `"about [term]"`, `"mentioning [term]"`

#### Author Filters

- `"by author [name]"`

#### Source Filters

- `"from source [name]"`, `"from [platform]"`, `"on [platform]"`

### Batch Processing

#### Process Multiple Queries

```bash
curl -X POST "http://localhost/api/v1/batch_process" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "Investment opportunities in renewable energy",
      "Cryptocurrency market trends",
      "AI developments in finance"
    ],
    "force_refresh": false,
    "skip_embeddings": false,
    "chunk_batch_size": 5,
    "summary_batch_size": 3,
    "max_workers": 4
  }'
```

**Response**:
```json
{
  "batch_id": "batch_1654321098_abcd",
  "results": [
    {
      "query": "Investment opportunities in renewable energy",
      "summary": "Recent discussions across social media platforms indicate...",
      "chunks": [...]
    },
    ...
  ],
  "metadata": {
    "total_time_ms": 3456.78,
    "avg_time_per_query_ms": 1152.26,
    "queries_processed": 3,
    "timestamp": "2023-06-01T12:34:56.789Z"
  }
}
```

### Task Status Monitoring

#### Check Task Status

```bash
curl -X GET "http://localhost/api/v1/batch_status/{task_id}"
```

**Response (Processing)**:
```json
{
  "status": "processing",
  "message": "The task is being processed. Please check back soon.",
  "task_id": "query_1654321098_abcd",
  "created_at": "2023-06-01T12:34:56.789Z"
}
```

**Response (Completed)**:
```json
{
  "status": "completed",
  "result": {
    "chunks": [...],
    "summary": "Recent discussions across social media platforms indicate...",
    "metadata": {...}
  }
}
```

**Response (Expired)**:
```json
{
  "status": "expired",
  "message": "Task query_1654321098_abcd was completed but results have expired. Results are only kept for 10 minutes.",
  "completed_at": "2023-06-01T12:34:56.789Z",
  "task_id": "query_1654321098_abcd"
}
```

### Admin Endpoints

#### Trigger Cleanup

```bash
curl -X POST "http://localhost/api/v1/admin/cleanup?force=false"
```

#### Get Metrics

```bash
curl -X GET "http://localhost/api/v1/metrics"
```

#### Debug Routes

```bash
curl -X GET "http://localhost/api/v1/debug/routes"
```

## Monitoring and Troubleshooting

### Checking Container Status

```bash
# Docker
docker-compose -f deployment/docker-compose.yml ps

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Restart application
docker-compose -f deployment/docker-compose.yml restart

# Stop application
docker-compose -f deployment/docker-compose.yml down
```

### Environment Detection Issues

**Problem**: Container logs show PostgreSQL connection errors, or environment detected as 'replit' instead of 'docker'.

**Solution**: The system now includes robust environment detection fixes:
- Explicit environment variables in `docker-compose.yml`
- Enhanced `DataConfig` to properly pass environment information
- Fixed hard-coded storage creation calls

**Verification**:
```bash
docker exec <container_id> python -c "
from config.env_loader import detect_environment
from config.storage import StorageFactory
from knowledge_agents.data_ops import DataConfig

config = DataConfig.from_config()
print('Environment:', detect_environment())
print('DataConfig env:', config.env)
storage = StorageFactory.create(config, config.env)
print('Storage type:', type(storage['complete_data']).__name__)
"
```

**Expected output**:
- Environment: docker
- DataConfig env: docker
- Storage type: FileCompleteDataStorage

### Data Loading Delays

If health checks fail during initial startup:

1. Verify S3 connectivity and credentials
2. Check logs for data loading progress
3. Consider increasing health check `start_period` and `retries`
4. Ensure the application responds with appropriate status during initialization
5. Monitor data processing logs to track progress

### Common Issues

#### Missing AWS Credentials

The application now skips S3 client initialization when AWS credentials are missing and logs a warning instead of failing. Provide the full set of variables in `.env` to enable cloud synchronization.

#### Test Failures

If tests fail during startup:
1. Check test logs in `test_results/`
2. Verify AWS credentials
3. Check for network connectivity issues
4. Validate data access permissions

#### Performance Issues

If the application performs poorly:
1. Check resource allocation in `docker-compose.yml`
2. Monitor CPU and memory usage
3. Verify data volume performance
4. Adjust worker count and resource limits

## Refresh Dashboard

The Knowledge Agent includes a web-based refresh dashboard for monitoring and controlling automated data refreshes.

### Accessing the Dashboard

- **Docker**: `http://localhost/refresh`
- **Replit**: `https://your-app.replit.app/refresh`

### Dashboard Features

- **System Status**: Real-time status of refresh operations
- **Performance Metrics**: Total runs, current row count, success rate, average duration
- **Controls**: Start/stop auto-refresh, run one-time refresh, update interval
- **Activity Log**: Recent refresh activity

### Dashboard API Endpoints

All endpoints are under `/refresh/api`:

```bash
# Get status
curl -X GET "http://localhost/refresh/api/status"

# Get metrics
curl -X GET "http://localhost/refresh/api/metrics"

# Control refresh
curl -X POST "http://localhost/refresh/api/control" \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
# Actions: "start", "stop", "refresh_once"

# Update configuration
curl -X POST "http://localhost/refresh/api/config" \
  -H "Content-Type: application/json" \
  -d '{"interval_seconds": 3600, "max_retries": 3}'
```

### Security

Protect control endpoints with a shared secret by setting `REFRESH_CONTROL_TOKEN`:

```bash
# Environment variable
REFRESH_CONTROL_TOKEN=your_secret_token

# Usage in URL
curl "http://localhost/refresh/api/status?token=your_secret_token"

# Usage in header
curl -X POST "http://localhost/refresh/api/control" \
  -H "X-Refresh-Token: your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
```

### CLI Control

Control the refresh manager from the command line:

```bash
# Check status
python scripts/refresh_control.py status

# Start auto-refresh
python scripts/refresh_control.py start --interval 3600

# Stop auto-refresh
python scripts/refresh_control.py stop

# Run one-time refresh
python scripts/refresh_control.py run-once

# With custom base URL
python scripts/refresh_control.py status --base http://host/refresh/api
```

### Auto-Start Configuration

Deployments start the refresh manager automatically by default. Configure via environment variables:

```bash
# Disable auto-start
AUTO_REFRESH_MANAGER=false

# Custom interval (seconds)
DATA_REFRESH_INTERVAL=7200  # 2 hours
```

## Task Management

The Knowledge Agent includes a robust task management system:

1. **Tracks Background Tasks**: Maintains status information for all background tasks
2. **Persists Task History**: Stores task history in `batch_history.json` file in the logs directory
3. **Provides Detailed Status**: Offers detailed status information through `/api/v1/batch_status/{task_id}` endpoint
4. **Performs Automatic Cleanup**: Periodically removes old task results to prevent memory leaks
5. **Preserves Task History**: Maintains a record of completed tasks even after results are removed from memory

This system ensures that users can track the status of their queries even if they were submitted hours ago, while preventing memory issues from accumulating task results.

## Success Indicators

### Deployment Success

- ✅ Health check passes within 10 seconds
- ✅ Root endpoint (`/`) returns `{"status": "ok"}`
- ✅ No "port not opening" errors
- ✅ Environment correctly detected (docker or replit)

### Initialization Success

- ✅ Background logs show "Background initialization completed successfully!"
- ✅ `/api/v1/initialization-status` shows services are available
- ✅ Database schema is created (Replit)
- ✅ Data files exist (Docker)

### Data Processing Success

- ✅ `/api/v1/initialization-status` shows `"ready": true`
- ✅ Query endpoints return results
- ✅ Data files exist or database has records
- ✅ Embeddings health check shows healthy status

## Maintenance

### Data Wipe Utilities

Use with extreme caution:

```bash
# Development (Replit): wipe KV, Object Storage, PostgreSQL, and files
python scripts/wipe_all_data.py --yes

# Production DB by full DSN
python scripts/wipe_all_data.py --yes --database-url "postgres://user:pass@host:5432/db"

# Production DB by discrete params
python scripts/wipe_all_data.py --yes --pg-host host --pg-user user --pg-password pass

# Skip specific scopes
python scripts/wipe_all_data.py --yes --no-kv --no-objects
python scripts/wipe_all_data.py --yes --no-files
```

## Emergency Rollback

If deployment fails completely:

1. **Revert to minimal startup**:
   ```bash
   # Docker: modify docker-compose.yml command
   command: ["python3", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "80"]
   ```

2. **Disable all background processing**:
   ```bash
   AUTO_CHECK_DATA=false
   AUTO_PROCESS_DATA_ON_INIT=false
   AUTO_REFRESH_MANAGER=false
   ENABLE_DATA_SCHEDULER=false
   ```

3. **Use minimal initialization**:
   ```bash
   # Replit: modify .replit run command
   run = "mkdir -p logs data temp_files && python -m uvicorn api.app:app --host 0.0.0.0 --port 80 --log-level info"
   ```

This should get a basic server running for debugging.

---

For detailed API endpoint documentation, see [`api/README_REQUESTS.md`](../api/README_REQUESTS.md).

