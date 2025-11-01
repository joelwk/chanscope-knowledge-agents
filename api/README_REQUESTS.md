# Knowledge Agent API Documentation

This document provides detailed information about the Knowledge Agent API endpoints, including request formats, parameters, and example usage.

## Table of Contents

- [Overview](#overview)
- [Base URLs](#base-urls)
- [Health Check Endpoints](#health-check-endpoints)
- [Query Processing Endpoints](#query-processing-endpoints)
- [Batch Processing Endpoints](#batch-processing-endpoints)
- [Data Management Endpoints](#data-management-endpoints)
- [Embedding Management Endpoints](#embedding-management-endpoints)
- [Debug Endpoints](#debug-endpoints)
- [Error Handling](#error-handling)
- [Natural Language Database Query](#natural-language-database-query)

## Overview

The Knowledge Agent API provides a comprehensive set of endpoints for querying and analyzing data from multiple social media platforms. The API follows RESTful principles and returns responses in JSON format.

## Base URLs

API version 1 endpoints are available under the `/api/v1` path.

**Note**: For deployment-specific base URLs and configuration, see [`deployment/DEPLOYMENT.md`](../deployment/DEPLOYMENT.md).

## Health Check Endpoints

### Basic Health Check

```bash
# Check if the API is running
curl -X GET "http://localhost/api/health"
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Service is running",
  "timestamp": "2023-06-01T12:34:56.789Z",
  "environment": {
    "docker_env": true,
    "service_type": "api",
    "api_version": "1.0.0",
    "is_replit": false
  },
  "data_status": {
    "root_data_exists": true,
    "stratified_data_exists": true,
    "logs_exists": true
  }
}
```

### Replit-Specific Health Check

```bash
# Check Replit-specific environment details
curl -X GET "http://localhost/api/health_replit"
```

### S3 Connection Health Check

```bash
# Check S3 connection and bucket access
curl -X GET "http://localhost/api/health/s3"
```

**Response:**
```json
{
  "s3_status": "connected",
  "bucket_access": true,
  "bucket_name": "your-bucket-name",
  "bucket_details": {
    "prefix": "unified/",
    "region": "us-east-1",
    "has_contents": true
  },
  "aws_region": "us-east-1",
  "latency_ms": 123.45
}
```

### Provider Health Check

```bash
# Check health of a specific provider (openai, grok, venice)
curl -X GET "http://localhost/api/health/provider/openai"
```

**Response:**
```json
{
  "status": "healthy",
  "provider": "openai",
  "models_available": 42,
  "latency_ms": 234.56
}
```

### All Providers Health Check

```bash
# Check health of all configured providers
curl -X GET "http://localhost/api/health/all"
```

### Cache Health Check

```bash
# Check health of the in-memory cache
curl -X GET "http://localhost/api/health/cache"
```

**Response:**
```json
{
  "status": "healthy",
  "type": "in_memory",
  "metrics": {
    "hit_ratio": "75.50%",
    "hits": 755,
    "misses": 245,
    "errors": 0,
    "total_requests": 1000
  },
  "configuration": {
    "enabled": true,
    "ttl": 3600
  }
}
```

### Embeddings Health Check

```bash
# Check health of embeddings
curl -X GET "http://localhost/api/health/embeddings"
```

**Response:**
```json
{
  "status": "healthy",
  "issues": [],
  "metrics": {
    "coverage_percentage": 99.8,
    "total_records": 10000,
    "records_with_embeddings": 9980,
    "dimension_mismatches": 0,
    "embedding_dimensions": 1536,
    "is_mock_data": false
  },
  "timestamp": "2023-06-01T12:34:56.789Z"
}
```

## Query Processing Endpoints

### Process a Query

```bash
# Process a query synchronously
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "force_refresh": false,
    "skip_embeddings": false,
    "use_background": false
  }'
```

**Response:**
```json
{
  "status": "completed",
  "task_id": "query_1654321098_abcd",
  "chunks": [
    {
      "text": "Solar energy investments are projected to grow by 25% in the next year...",
      "score": 0.92,
      "metadata": {
        "timestamp": "2023-05-15T08:45:12Z",
        "source": "4chan",
        "thread_id": "12345678"
      }
    },
    {
      "text": "Wind power capacity is expected to double in the next five years...",
      "score": 0.87,
      "metadata": {
        "timestamp": "2023-05-16T14:22:33Z",
        "source": "X",
        "thread_id": "12345679"
      }
    }
  ],
  "summary": "Recent discussions across social media platforms indicate growing interest in renewable energy investments, particularly in solar and wind power. Several users have highlighted the potential for significant growth in these sectors, with solar energy investments projected to grow by 25% in the next year and wind power capacity expected to double in the next five years. There are also mentions of government incentives and tax credits making these investments more attractive.",
  "metadata": {
    "processing_time_ms": 1234.56,
    "num_relevant_strings": 5,
    "temporal_context": {
      "earliest_date": "2023-05-15T08:45:12Z",
      "latest_date": "2023-05-16T14:22:33Z"
    }
  }
}
```

### Process a Query in the Background

```bash
# Process a query asynchronously in the background
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "force_refresh": false,
    "skip_embeddings": false,
    "use_background": true
  }'
```

**Response:**
```json
{
  "status": "processing",
  "task_id": "query_1654321098_abcd",
  "message": "Query processing started in background"
}
```

### Check Query Status

```bash
# Check the status of a background query
curl -X GET "http://localhost/api/v1/batch_status/query_1654321098_abcd"
```

**Response (Processing):**
```json
{
  "status": "processing",
  "message": "The task is being processed. Please check back soon.",
  "task_id": "query_1654321098_abcd",
  "created_at": "2023-06-01T12:34:56.789Z"
}
```

**Response (Completed):**
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

**Response (Expired):**
```json
{
  "status": "expired",
  "message": "Task query_1654321098_abcd was completed but results have expired. Results are only kept for 10 minutes.",
  "completed_at": "2023-06-01T12:34:56.789Z",
  "task_id": "query_1654321098_abcd"
}
```

### Process Recent Query

```bash
# Process a query using recent data from the last 6 hours
curl -X GET "http://localhost/api/v1/process_recent_query?select_board=biz"
```

**Response:**
```json
{
  "status": "processing",
  "task_id": "query_1654321098_abcd",
  "message": "Query processing started in background",
  "time_range": {
    "start": "2023-06-01T06:34:56.789Z",
    "end": "2023-06-01T12:34:56.789Z"
  }
}
```

## Batch Processing Endpoints

### Process Multiple Queries

```bash
# Process multiple queries in a batch
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

**Response:**
```json
{
  "batch_id": "batch_1654321098_abcd",
  "results": [
    {
      "query": "Investment opportunities in renewable energy",
      "summary": "Recent discussions across social media platforms indicate...",
      "chunks": [...]
    },
    {
      "query": "Cryptocurrency market trends",
      "summary": "Analysis of cryptocurrency discussions across platforms reveals...",
      "chunks": [...]
    },
    {
      "query": "AI developments in finance",
      "summary": "Discussions about AI in finance on various platforms highlight...",
      "chunks": [...]
    }
  ],
  "metadata": {
    "total_time_ms": 3456.78,
    "avg_time_per_query_ms": 1152.26,
    "queries_processed": 3,
    "timestamp": "2023-06-01T12:34:56.789Z"
  }
}
```

## Data Management Endpoints

### Stratify Data

```bash
# Trigger data stratification
curl -X POST "http://localhost/api/stratify"
```

**Response:**
```json
{
  "status": "success",
  "message": "Stratification completed successfully",
  "data": {
    "stratified_rows": 10000,
    "stratified_file": "/app/data/stratified/stratified_sample.csv"
  }
}
```

## Embedding Management Endpoints

### Trigger Embedding Generation

```bash
# Trigger background embedding generation
curl -X POST "http://localhost/api/trigger_embedding_generation"
```

**Response:**
```json
{
  "status": "started",
  "message": "Embedding generation started in background",
  "task_id": "embedding_generation"
}
```

### Check Embedding Generation Status

```bash
# Check the status of embedding generation
curl -X GET "http://localhost/api/embedding_status"
```

**Response (Running):**
```json
{
  "status": "running",
  "message": "Embedding generation is in progress",
  "progress": 45,
  "start_time": "2023-06-01T12:34:56.789Z",
  "total_rows": 10000
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "message": "Embedding generation completed successfully",
  "progress": 100,
  "start_time": "2023-06-01T12:34:56.789Z",
  "end_time": "2023-06-01T12:45:12.345Z",
  "total_rows": 10000
}
```

## Debug Endpoints

### List All Routes

```bash
# Get a list of all registered routes
curl -X GET "http://localhost/api/debug/routes"
```

**Response:**
```json
{
  "routes": [
    {
      "endpoint": "health_check",
      "methods": ["GET"],
      "path": "/api/health",
      "is_api": true
    },
    {
      "endpoint": "base_query",
      "methods": ["POST"],
      "path": "/api/v1/query",
      "is_api": true
    }
  ],
  "total_routes": 25,
  "api_routes": 20,
  "other_routes": 5
}
```

### Debug Request

```bash
# Get details about the current request
curl -X GET "http://localhost/api/debug/request"
```

**Response:**
```json
{
  "method": "GET",
  "path": "/api/debug/request",
  "headers": {
    "host": "localhost",
    "user-agent": "curl/7.68.0",
    "accept": "*/*"
  },
  "query_params": {}
}
```

## Error Handling

The API uses consistent error responses across all endpoints. Error responses include:

- `status_code`: HTTP status code
- `detail`: Object containing error details
  - `message`: Human-readable error message
  - `error_code`: Machine-readable error code
  - `details`: Additional error details (when available)

Example error response:

```json
{
  "detail": {
    "message": "Provider error for openai",
    "error_code": "PROVIDER_ERROR",
    "details": {
      "provider": "openai",
      "error": "API key not valid",
      "latency_ms": 123.45
    }
  }
}
```

Common error codes:

- `VALIDATION_ERROR`: Invalid request parameters
- `PROVIDER_ERROR`: Error with a model provider
- `PROCESSING_ERROR`: Error during query processing
- `CONFIGURATION_ERROR`: Error in system configuration
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `INTERNAL_SERVER_ERROR`: Unexpected server error

## Advanced Usage

### Custom Task IDs

You can provide your own task ID when submitting a query:

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "use_background": true,
    "task_id": "my_custom_task_id_123"
  }'
```

### Source-Specific Queries

Filter queries to a specific source:

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "select_board": "biz"
  }'
```

### Date-Filtered Queries

Filter queries to a specific date range:

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "filter_date": "2023-05-15"
  }'
```

### Provider Selection

Specify which providers to use for different operations:

```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "embedding_provider": "openai",
    "chunk_provider": "venice",
    "summary_provider": "grok"
  }'
```

## Natural Language Database Query

### `POST /api/v1/nl_query`

Process a natural language query against the database using LLM-generated SQL. This endpoint converts natural language like "Show me posts from the last hour" into SQL and executes it using a two-stage LLM architecture.

#### Request Body

```json
{
  "query": "Give me threads from the last hour",
  "limit": 100,
  "provider": "openai"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Natural language query to execute |
| limit | integer | No | Maximum number of results to return (default: 100) |
| provider | string | No | Model provider to use for SQL generation (default: system default) |

> **Note:** It's recommended to use the `limit` parameter to control result size, especially for broad queries without specific filters that could return large result sets. Consider using a more specific time filter (e.g., "last hour" instead of "last month") when searching for common terms.

#### Response

```json
{
  "status": "success",
  "query": "Give me threads from the last hour",
  "description": {
    "original_query": "Give me threads from the last hour",
    "query_time": "2023-08-01T12:34:56.789Z",
    "filters": ["Time: Last hour"],
    "time_filter": "Last hour"
  },
  "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC LIMIT %s",
  "record_count": 15,
  "data": [
    {
      "id": 1234,
      "thread_id": "abc123",
      "content": "Example post content",
      "posted_date_time": "2023-08-01T12:30:00Z",
      "source": "X",
      "author": "username"
    },
    // ...more records
  ],
  "execution_time_ms": 123.45,
  "metadata": {
    "processing_time_ms": 123.45,
    "sql_generation_method": "llm",
    "timestamp": "2023-08-01T12:34:56.789Z",
    "provider": "openai"
  }
}
```

#### How It Works

The system uses a two-stage LLM approach:

1. **SQL Generation**: Your query is passed to an LLM to convert natural language to SQL
2. **SQL Validation**: A second step validates the SQL for security and correctness
3. **Parameter Extraction**: Parameters are automatically extracted from your query
4. **Query Execution**: The final SQL is executed against the PostgreSQL database

For common query patterns, a hybrid approach is used that matches templates first before falling back to the LLM, providing both speed and flexibility.

#### Example Queries

- `"Give me threads from the last hour"`
- `"Show posts from yesterday containing crypto"`
- `"Find messages from the last 3 days by author john"`
- `"Get content from source X about AI from this week"`
- `"Show messages containing machine learning from this month"`

#### Supported Time Filters

- Last hour: `"last hour"`, `"past hour"`, `"previous hour"`, `"recent hour"`
- Last X hours: `"last 5 hours"`, `"past 12 hours"`, etc.
- Today: `"today"`, `"this day"`
- Yesterday: `"yesterday"`, `"previous day"`
- Last X days: `"last 7 days"`, `"past 30 days"`, etc.
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

#### Advanced Usage

The LLM-based approach allows for more complex queries:

- **Combinations**: `"Find posts from source X about AI by author john from last week"`
- **Contextual Understanding**: `"Show me recent discussions about the latest crypto regulations"`
- **Natural Phrasing**: `"What are people saying about market trends this month?"`

#### Errors

- 400: Invalid query format or couldn't parse natural language
- 500: Database error or other server-side issue

> **Note**: This endpoint requires PostgreSQL and is only available in Replit environments. See [`deployment/DEPLOYMENT.md`](../deployment/DEPLOYMENT.md) for deployment information.

## Creative Use Cases

This section demonstrates advanced usage patterns and creative examples that showcase the API's capabilities.

### 1. Natural Language Database Queries with Complex Filters

#### Real-Time Market Monitoring

Query posts about specific topics within narrow time windows:

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts about Bitcoin from the last 3 hours containing mentions of ETFs",
    "limit": 50,
    "format_for_llm": true
  }'
```

#### Multi-Dimensional Content Analysis

Combine multiple filters for precise data extraction:

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find threads from biz board by author john about crypto regulations from last week",
    "limit": 100
  }'
```

#### Trend Analysis Queries

Understand what people are discussing about specific topics:

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are people saying about AI regulation this month?",
    "limit": 200
  }'
```

#### Time-Series Data Extraction

Extract data for specific time periods with content filters:

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts from yesterday containing machine learning discussions",
    "limit": 150
  }'
```

### 2. Batch Processing Workflows

#### Parallel Query Processing

Process multiple related queries simultaneously:

```bash
curl -X POST "http://localhost/api/v1/batch_process" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "Investment opportunities in renewable energy",
      "Cryptocurrency market trends",
      "AI developments in finance",
      "Regulatory changes affecting tech companies"
    ],
    "force_refresh": false,
    "chunk_batch_size": 5,
    "summary_batch_size": 3,
    "max_workers": 4
  }'
```

#### Custom Task Tracking

Use custom task IDs for tracking specific workflows:

```bash
# Submit queries with custom task IDs
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Market sentiment analysis",
    "task_id": "market_analysis_1",
    "use_background": false
  }'

# Check status later
curl -X GET "http://localhost/api/v1/batch_status/market_analysis_2024_01_15"
```

#### Batch Status Monitoring Pattern

Monitor multiple batch jobs:

```bash
# Submit batch
BATCH_ID=$(curl -X POST "http://localhost/api/v1/batch_process" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["query1", "query2", "query3"]}' | jq -r '.batch_id')

# Poll for completion
while true; do
  STATUS=$(curl -s "http://localhost/api/v1/batch_status/$BATCH_ID" | jq -r '.status')
  if [ "$STATUS" = "completed" ]; then
    echo "Batch completed!"
    curl -s "http://localhost/api/v1/batch_status/$BATCH_ID" | jq '.result'
    break
  fi
  sleep 5
done
```

### 3. Background Task Management

#### Asynchronous Query Processing

Submit long-running queries for background processing:

```bash
# Submit query for background processing
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Comprehensive analysis of financial market trends",
    "use_background": true
  }'

# Response: {"status": "processing", "task_id": "query_1234567890_abcd"}
```

#### Status Polling Pattern

Implement efficient status polling:

```bash
TASK_ID="query_1234567890_abcd"

# Poll with exponential backoff
DELAY=2
MAX_DELAY=30
while true; do
  RESPONSE=$(curl -s "http://localhost/api/v1/batch_status/$TASK_ID")
  STATUS=$(echo $RESPONSE | jq -r '.status')
  
  case $STATUS in
    "completed")
      echo "Query completed!"
      echo $RESPONSE | jq '.result'
      break
      ;;
    "failed")
      echo "Query failed!"
      echo $RESPONSE | jq '.error'
      break
      ;;
    "expired")
      echo "Query expired"
      break
      ;;
    *)
      echo "Status: $STATUS. Waiting ${DELAY}s..."
      sleep $DELAY
      DELAY=$((DELAY < MAX_DELAY ? DELAY * 2 : MAX_DELAY))
      ;;
  esac
done
```

#### Task Result Retrieval

Retrieve results with metadata:

```bash
# Get task status with full result
curl -X GET "http://localhost/api/v1/batch_status/query_1234567890_abcd" | jq '{
  status: .status,
  chunks_count: (.result.chunks | length),
  summary: .result.summary,
  processing_time: .result.metadata.processing_time_ms
}'
```

### 4. Multi-Provider Query Strategies

#### Provider-Specific Queries

Use different providers for different operations:

```bash
# Use OpenAI for embeddings, Grok for summaries
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Market analysis",
    "embedding_provider": "openai",
    "chunk_provider": "venice",
    "summary_provider": "grok"
  }'
```

#### Provider Health Monitoring

Monitor provider availability before processing:

```bash
# Check all providers
curl -X GET "http://localhost/api/v1/health/all"

# Check specific provider
curl -X GET "http://localhost/api/v1/health/provider/openai"
curl -X GET "http://localhost/api/v1/health/provider/grok"
curl -X GET "http://localhost/api/v1/health/provider/venice"
```

### 5. Refresh Dashboard Integration

#### Automated Refresh Control

Programmatically control the refresh dashboard:

```bash
# Start auto-refresh with custom interval
curl -X POST "http://localhost/refresh/api/config" \
  -H "Content-Type: application/json" \
  -H "X-Refresh-Token: your_token" \
  -d '{
    "interval_seconds": 7200,
    "max_retries": 3
  }'

# Trigger refresh
curl -X POST "http://localhost/refresh/api/control" \
  -H "Content-Type: application/json" \
  -H "X-Refresh-Token: your_token" \
  -d '{"action": "start"}'
```

#### Metrics Monitoring

Monitor refresh performance:

```bash
# Get current metrics
curl -X GET "http://localhost/refresh/api/metrics" | jq '{
  total_runs: .total_runs,
  current_rows: .current_row_count,
  success_rate: .success_rate,
  avg_duration: .average_duration
}'

# Get status
curl -X GET "http://localhost/refresh/api/status" | jq '{
  is_running: .is_running,
  current_job: .current_job,
  next_run: .next_run
}'
```

### 6. Comprehensive Health Monitoring

#### Health Check Workflow

Implement comprehensive health monitoring:

```bash
# Check all systems
HEALTH_CHECKS=(
  "/api/v1/health"
  "/api/v1/health/s3"
  "/api/v1/health/cache"
  "/api/v1/health/embeddings"
  "/api/v1/health/all"
)

for endpoint in "${HEALTH_CHECKS[@]}"; do
  echo "Checking $endpoint..."
  curl -s "http://localhost$endpoint" | jq '.status'
done
```

#### Cache Performance Monitoring

Monitor cache effectiveness:

```bash
curl -X GET "http://localhost/api/v1/health/cache" | jq '{
  hit_ratio: .metrics.hit_ratio,
  total_requests: .metrics.total_requests,
  errors: .metrics.errors
}'
```

#### Embedding Health Monitoring

Monitor embedding coverage and quality:

```bash
curl -X GET "http://localhost/api/v1/health/embeddings" | jq '{
  status: .status,
  coverage: .metrics.coverage_percentage,
  total_records: .metrics.total_records,
  issues: .issues
}'
```

### 7. Advanced Query Patterns

#### Source-Specific Analysis

Analyze data from specific sources:

```bash
# Filter to specific board
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Financial market trends",
    "select_board": "biz"
  }'
```

#### Temporal Analysis

Analyze data within specific time ranges:

```bash
# Query with date filter
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cryptocurrency discussions",
    "filter_date": "2024-01-15"
  }'
```

#### Recent Data Processing

Process queries using only recent data:

```bash
# Process recent query (last 6-12 hours)
curl -X GET "http://localhost/api/v1/process_recent_query?select_board=biz&use_background=true"
```

### 8. Error Handling Patterns

#### Comprehensive Error Handling

Handle various error scenarios:

```bash
# Submit query with error handling
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

case $HTTP_CODE in
  200|201)
    echo "Success: $BODY" | jq '.'
    ;;
  400)
    echo "Bad Request: $BODY" | jq '.detail'
    ;;
  500)
    echo "Server Error: $BODY" | jq '.detail'
    ;;
  *)
    echo "Unexpected status: $HTTP_CODE"
    ;;
esac
```

#### Task Expiration Handling

Handle expired task results:

```bash
STATUS=$(curl -s "http://localhost/api/v1/batch_status/$TASK_ID" | jq -r '.status')

if [ "$STATUS" = "expired" ]; then
  echo "Task expired. Check batch_history.json for historical records."
  # Optionally, resubmit the query
fi
```

## Best Practices

1. **Use Background Processing**: For long-running queries, use `use_background: true` to avoid timeouts
2. **Monitor Task Status**: Implement polling with exponential backoff for background tasks
3. **Set Appropriate Limits**: Use `limit` parameter in NL queries to prevent large result sets
4. **Handle Errors Gracefully**: Implement comprehensive error handling for all API calls
5. **Monitor Health**: Regularly check health endpoints to ensure system availability
6. **Use Custom Task IDs**: For tracking workflows, use custom task IDs
7. **Cache When Appropriate**: Leverage the built-in cache for frequently accessed queries
8. **Monitor Embeddings**: Regularly check embedding health to ensure data quality 