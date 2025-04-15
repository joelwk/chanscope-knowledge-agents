# Chanscope: Biologically-Inspired 4chan Analysis System

## Overview

Chanscope is an advanced data ingestion, embedding, and generative query pipeline specifically designed for comprehensive 4chan data analysis. Inspired by biological systems, Chanscope orchestrates multi-stage processing workflows to extract actionable insights from large volumes of thread data using stratified sampling, distributed embedding generation, and context-aware summarization.

The system provides a robust API layer that can be integrated with autonomous AI agents and agentic systems, enabling temporal analysis, cross-source verification, and predictive analytics through a biologically-inspired architecture that mimics natural information processing systems.

## Key Features & Biological Parallels

- **Adaptive Data Ingestion**: Like organisms absorbing nutrients selectively, Chanscope ingests data from S3 storage with configurable retention periods and temporal filtering.

- **Stratified Sampling**: Similar to how biological systems extract relevant signals from noise, the system performs intelligent stratification of raw data to maximize insight from minimal processing.

- **Dynamic Embedding Generation**: Analogous to neural encoding in biological systems, Chanscope transforms text data into vector representations stored in optimized `.npz` format.

- **Homeostatic Refresh Mechanisms**: The system maintains data freshness through feedback loops that trigger refresh operations only when needed, mimicking homeostasis in living systems.

- **Multi-agent Query Processing**: Distributed processing of queries across specialized components resembles cellular specialization in complex organisms.

- **Enhanced Natural Language Query Processing**: The NL query endpoint now incorporates a robust validation mechanism that retains the original query context in the SQL generation process, ensuring that essential filters (particularly content filters) are applied accurately. This mirrors natural quality control in biological systems.

- **Environmental Adaptation**: Automatic environment detection and configuration adjustment similar to how organisms adapt to their surroundings.

## System Architecture

Chanscope's architecture follows a biologically-inspired pattern with distinct yet interconnected processing stages:

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Data Sources  │         │  Processing Core  │         │  Query System   │
│  ┌────────────┐ │         │  ┌────────────┐  │         │ ┌────────────┐  │
│  │    S3      │◄├─┐       │  │ Stratified │  │     ┌───┼►│   Query    │  │
│  │  Storage   │ │ │       │  │  Sampling  │  │     │   │ │ Processing │  │
│  └────────────┘ │ │       │  └─────┬──────┘  │     │   │ └─────┬──────┘  │
└─────────────────┘ │       │        │         │     │   │       │         │
                    │       │  ┌─────▼──────┐  │     │   │ ┌─────▼──────┐  │
┌─────────────────┐ │       │  │ Embedding  │  │     │   │ │   Chunk    │  │
│  Memory System  │ │       │  │ Generation │  │     │   │ │ Processing │  │
│  ┌────────────┐ │ │       │  └─────┬──────┘  │     │   │ └─────┬──────┘  │
│  │ Complete   │◄┼─┘       │        │         │     │   │       │         │
│  │    Data    │ │         │        │         │     │   │ ┌─────▼──────┐  │
│  └────────────┘ │         │        │         │     │   │ │   Final    │  │
│  ┌────────────┐ │         │        │         │     │   │ │ Summarizer │  │
│  │ Stratified │◄├─────────┼────────┘         │     │   │ └────────────┘  │
│  │   Sample   │ │         │                  │     │   │                 │
│  └────────────┘ │         │                  │     │   └─────────────────┘
│  ┌────────────┐ │         │                  │     │
│  │ Embeddings │◄├─────────┼──────────────────┼─────┘
│  │   (.npz)   │ │         │                  │
│  └────────────┘ │         └──────────────────┘
└─────────────────┘
```

### Processing Pipeline

1. **Data Ingestion**: Retrieves data from S3 starting from `DATA_RETENTION_DAYS` ago.
2. **Stratification**: Samples the complete dataset using `sampler.py` to create a representative subset.
3. **Embedding Generation**: Creates embeddings stored in `.npz` format with thread ID mappings.
4. **Query Processing**: Leverages embeddings for semantic search and incorporates an enhanced natural language query processing module. This module uses LLMSQLGenerator to convert natural language queries into SQL queries while preserving the original query context and enforcing essential filters (particularly content filters). This approach mirrors biological quality control mechanisms, ensuring precise data retrieval.

## Quick Start Guide

Chanscope supports multiple deployment environments with environment-specific optimizations:

### Docker Environment

```bash
# Clone repository and navigate to project directory
git clone https://github.com/your-org/chanscope.git
cd chanscope

# Build and run containers
docker-compose -f deployment/docker-compose.yml build
docker-compose -f deployment/docker-compose.yml up -d

# Check system status
curl http://localhost/api/v1/health/all

# Run a basic query
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze recent cryptocurrency trends"}'
```

### Replit Environment

```bash
# Fork the Replit project: https://replit.com/@jwkonitzer/chanscope-knowledge-agents

# The .replit file will automatically run deployment/setup.sh
# This sets up appropriate environment variables for Replit

# Check system status
curl https://chanscope-knowledge-agents.jwkonitzer.repl.co/api/v1/health/all

# Run a basic query
curl -X POST "https://chanscope-knowledge-agents.jwkonitzer.repl.co/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze recent cryptocurrency trends"}'
```

### Local Environment

```bash
# Clone repository and navigate to project directory
git clone https://github.com/your-org/chanscope.git
cd chanscope

# Set up Python environment
poetry install

# Load environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize data and start API server
poetry run python scripts/scheduled_update.py refresh
poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run a basic query
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze recent cryptocurrency trends"}'
```

## Detailed Usage

### Data Operations

```bash
# Standard refresh (processes all data but reuses existing stratified sample and embeddings)
# Note: This does NOT regenerate the stratified sample unless it's missing
poetry run python scripts/scheduled_update.py refresh

# Force complete refresh (regenerates stratified sample and embeddings)
# This ensures your stratified sample and embeddings reflect the latest data
poetry run python scripts/scheduled_update.py refresh --force-refresh

# Two-stage refresh (sample now, embeddings later)
# Analogous to staged biological processes
poetry run python scripts/scheduled_update.py refresh --force-refresh --skip-embeddings
poetry run python scripts/scheduled_update.py embeddings

# Continuous scheduled refresh with forced stratification regeneration
# Ensures both data processing and stratified sample stay current
poetry run python scripts/scheduled_update.py refresh --continuous --force-refresh --interval=3600

# Check system status (data freshness, embedding status)
# Like monitoring an organism's vital signs
poetry run python scripts/scheduled_update.py status
```

> **Important**: The `--force-refresh` flag controls whether to regenerate the stratified sample and embeddings. Without this flag, existing stratified samples and embeddings will be reused even if they're outdated. The system always processes all data files regardless of this flag.

### Query Processing

Chanscope's query processing resembles a neural network's distributed processing:

```bash
# Standard query
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze cryptocurrency trends in relation to federal policy"
  }'

# Multi-provider query with fallback chain (biological redundancy)
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze cryptocurrency trends in relation to federal policy",
    "providers": {
      "summarization": ["openai", "grok", "venice"],
      "embedding": "openai",
      "chunk_generation": "openai"
    }
  }'

# Temporal analysis with date range (like studying evolutionary changes)
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Evolution of AI model capabilities discussion",
    "filter_date": "2025-03-15",
    "end_date": "2025-04-10"
  }'

# Background batch processing (asynchronous processing like biological background processes)
curl -X POST "http://localhost/api/v1/batch_process" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "Latest developments in AI regulation",
      "Public sentiment toward AI safety measures",
      "Corporate responses to regulation proposals"
    ],
    "task_id": "ai_regulation_analysis",
    "use_background": true
  }'
```

## Recommended Workflows

Chanscope supports distinct workflows for different stages of development and deployment:

### Testing Workflow

#### Docker Environment
```bash
# Run tests in isolation using dedicated testing compose file
docker-compose -f deployment/docker-compose.test.yml build
docker-compose -f deployment/docker-compose.test.yml up
```

#### Replit Environment
```bash
# Run tests in Replit using dedicated setup script
bash scripts/run_tests.sh --env=replit
```

#### Local Environment
```bash
# Run tests locally
bash scripts/run_tests.sh --env=local
```

### Production Deployment

#### Docker Environment
```bash
# Deploy application without running tests at startup
docker-compose -f deployment/docker-compose.yml build
docker-compose -f deployment/docker-compose.yml up -d
```

#### Replit Environment
```bash
# Deploy in Replit using the main setup script
bash deployment/setup.sh
# The .replit file is configured to run this automatically
```

## Monitoring & Maintenance

Chanscope provides comprehensive monitoring endpoints to track system health:

```bash
# Check overall API health with connection/provider tests
curl -X GET "http://localhost/api/v1/health/all"

# Check database connectivity and schema status
curl -X GET "http://localhost/api/v1/health/connections"

# Check S3 storage access and bucket status
curl -X GET "http://localhost/api/v1/health/s3"

# Check embedding coverage metrics
curl -X GET "http://localhost/api/v1/health/embeddings"

# View status of background tasks
curl -X GET "http://localhost/api/v1/batch_status/ai_regulation_analysis"
```

### Continuous Data Processing

For long-running deployments, enable the continuous update mode:

```bash
# Run as persistent service with hourly updates (like biological cycles)
poetry run python scripts/scheduled_update.py refresh --continuous --interval=3600
```

## Configuration & Environment Variables

Chanscope automatically detects the execution environment and configures itself accordingly, but you can override these settings through environment variables:

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `DATA_RETENTION_DAYS` | Number of days to retain data | `30` | Affects S3 data ingestion |
| `ENABLE_DATA_SCHEDULER` | Enable automatic data updates | `true` | Set to `false` for manual updates only |
| `SAMPLE_SIZE` | Size of stratified sample | `100000` | Adjust based on available memory |
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding generation | `10` | Lower for constrained environments |
| `FORCE_ENVIRONMENT` | Override environment detection | `null` | Options: `docker`, `replit`, `local` |
| `RUN_TESTS_ON_STARTUP` | Run tests during startup | `false` | Not recommended for production |

### Environment-Specific Considerations

#### Replit Environment
- Limited CPU, memory, and disk space
- Use smaller batch sizes and fewer workers
- Enable mock data for development/testing

#### Docker Environment
- Configure volumes for data persistence
- Set appropriate container resource limits
- Configure health checks with extended start periods

## Testing & Validation

Chanscope includes comprehensive test suites for validating system behavior:

```bash
# Run all tests with auto-detection of environment
scripts/run_tests.sh

# Run specific test categories
scripts/run_tests.sh --data-ingestion
scripts/run_tests.sh --embedding
scripts/run_tests.sh --endpoints
scripts/run_tests.sh --chanscope-approach
```

Testing validates key biological-inspired patterns:
- Data ingestion and processing pipelines (nutrient absorption)
- Feedback mechanisms for forced vs. non-forced refresh (homeostasis)
- Embedding generation and storage (neural encoding)
- Query processing and summarization (higher-order cognition)

## Contributing Guidelines

Contributions to Chanscope should follow these principles:

1. **Modular Design**: Each component should have well-defined interfaces, like specialized cells in an organism.
2. **Autonomous Operation**: Components should operate independently while coordinating through clear signaling mechanisms.
3. **Robust Error Handling**: Implement retry logic and circuit breakers, analogous to biological redundancy and repair mechanisms.
4. **Resource Efficiency**: Optimize for memory and processing constraints, similar to energy conservation in living systems.

When submitting changes:
1. Ensure tests pass in all environments
2. Follow existing code conventions
3. Document biological inspirations for architectural decisions
4. Add relevant tests for new features

## Future Work

Based on documented TODOs, planned enhancements include:

- **Monitoring Endpoints**: Enhanced API endpoints to monitor data freshness
- **Performance Optimization**: Parallel processing for embedding generation
- **Error Recovery Mechanisms**: Improved retry logic for S3 operations
- **Enhanced Testing**: Additional tests for data retention logic and embedding performance
- **Documentation Updates**: Comprehensive API documentation and operational guides

## License & Contact

[License Information]

For questions or contributions, please open an issue in the GitHub repository or contact the maintainers.