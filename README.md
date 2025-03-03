# Chanscope Knowledge Agent

## Overview
An advanced query system leveraging multiple AI providers (OpenAI, Grok, Venice) for comprehensive 4chan data analysis to extract actionable insights and patterns. The system provides a robust API layer that can be integrated with autonomous AI agents and agentic systems. It employs intelligent sampling techniques and a multi-stage analysis pipeline to process large volumes of 4chan data, enabling temporal analysis, cross-source verification, and predictive analytics.

### Platform Integration
- **Traditional Deployment**: Docker-based local or server deployment for controlled environments
- **[Replit Cloud](https://replit.com/@jwkonitzer/chanscope-knowledge-agents)**: 
  - Zero-setup cloud deployment with optimized performance
  - Automated hourly data updates
  - Environment-specific path handling
  - Optimized memory management
- **Agentic System Integration**: 
  The API architecture is designed to be integrated with autonomous AI agents and agentic systems, such as:
  - **[Virtuals Protocol](https://app.virtuals.io/prototypes/0x2cc92Fc77180815834FfdAa72C58f72d457C4308)/[X](https://x.com/4Chanscope)**: 
  The system can be integrated with autonomous AI agents through its API layer, enabling:
    - Consumption of 4chan data analysis through standardized API endpoints
    - Integration with agent memory systems for persistent context
    - Support for agent-driven data exploration and pattern recognition
    - Potential for onchain data validation and verification
    - Extensibility for custom agent-specific analysis patterns

### Core Architecture
- **Multi-Provider Architecture**
  - OpenAI (Primary): GPT-4o, text-embedding-3-large
  - Grok (Optional): grok-2-1212, grok-v1-embedding
  - Venice (Optional): dolphin-2.9.2-qwen2-72b, deepseek-r1-671b

- **Intelligent Data Processing**
  - Automated hourly data updates with incremental processing
  - Time-based and category-based stratified sampling
  - Board-specific data filtering and validation
  - Efficient large dataset handling with reservoir sampling
  - Automated data chunking and embedding generation
  - Cloud-optimized processing for Replit environment

- **Advanced Analysis Pipeline**
  - Real-time monitoring with 6-hour rolling window
  - Context-aware temporal analysis with validation
  - Parallel processing with automatic model fallback
  - Event mapping and relationship extraction
  - Cross-platform data synchronization
  - Enhanced S3 data streaming with board filtering
  - Optimized batch processing for query efficiency

- **API-First Design**
  - RESTful endpoints for all core functionality
  - Structured JSON responses for easy integration
  - Comprehensive error handling with detailed feedback
  - Batch processing for high-volume requests
  - Authentication and rate limiting for production use
  - Detailed documentation for third-party integration

For greater technical details and examples, refer to the [knowledge-agents](https://github.com/joelwk/knowledge-agents) repository.

## Analysis Capabilities

### 1. Temporal Analysis
- Thread dynamics tracking
- Activity burst detection
- Topic evolution mapping
- Cross-reference analysis

### 2. Signal Processing
- Source credibility rating
- Cross-mention validation
- Topic persistence assessment
- Impact measurement with confidence intervals

### 3. Pattern Detection
- Temporal sequence mapping
- Viral trigger identification
- Information flow tracking
- Anomaly detection

### 4. Metrics & Variables
- **Temporal**: timestamps, response times, activity frequency
- **Cascade**: thread depth, topic spread, lifetime
- **Content**: toxicity, relevance, uniqueness, influence
- **Forecast**: event probability, confidence bounds, reliability

## Integration with Agentic Systems

The Knowledge Agent is designed to serve as a backend for AI agents and agentic systems through its API layer:

### 1. Agent Integration Patterns
- **Direct API Consumption**: Agents can directly query the API endpoints
- **Memory Augmentation**: Results can be stored in agent memory systems
- **Decision Support**: Analysis can inform agent decision-making processes
- **Autonomous Monitoring**: Agents can set up scheduled queries for monitoring

### 2. Agent Capabilities Enabled
- **Contextual Understanding**: Deep understanding of 4chan discussions and trends
- **Pattern Recognition**: Identification of emerging patterns and anomalies
- **Temporal Awareness**: Understanding of how topics evolve over time
- **Cross-Reference Analysis**: Connecting related discussions across threads and boards

### 3. Implementation Examples
- Autonomous monitoring agents that alert on specific patterns
- Research agents that can explore and analyze specific topics
- Trend analysis agents that track evolving narratives
- Verification agents that cross-check claims across multiple sources

## Supported Models
The project supports multiple AI model providers:
- **OpenAI** (Required): Default provider for both completions and embeddings
  - Required: `OPENAI_API_KEY`
  - Models: `gpt-4o` (default), `text-embedding-3-large` (embeddings)
- **Grok (X.AI)** (Optional): Alternative provider with its own embedding model
  - Optional: `GROK_API_KEY`
  - Models: `grok-2-1212`, `grok-v1-embedding` (not publicly available)
- **Venice.AI** (Optional): Additional model provider for completion and chunking
  - Optional: `VENICE_API_KEY`
  - Models: `dolphin-2.9.2-qwen2-72b`, `deepseek-r1-671b`

## Quick Start

1. **Setup Environment**
```bash
git clone https://github.com/joelwk/knowledge-agents.git
cd knowledge-agents
cp .env.template .env  # Configure your API keys
```

2. **Required Environment Variables**
- `OPENAI_API_KEY`: Primary provider (Required)
- `GROK_API_KEY`: Alternative provider (Optional)
- `VENICE_API_KEY`: Additional provider (Optional)

3. **AWS Credentials**:
- `AWS_ACCESS_KEY_ID`: AWS access key ID (Required)
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key (Required)

  For AWS S3 access to 4chan data (hourly for the last 30 days), email chanscope@proton.me.

4. **Launch with Docker**
```bash
# Development
docker-compose -f deployment/docker-compose.dev.yml up --build -d

# Access Services
API: http://localhost:5000
Frontend: http://localhost:8000
```

## Replit Configuration

The project is configured to run seamlessly on Replit with optimized settings for the platform's environment.

### Replit Setup Files
- `.replit`: Configures the Replit environment
  - Defines run commands and entry points
  - Sets up port forwarding (5000 for API, 8000 for UI)
  - Configures Python environment and dependencies
  - Integrates with Replit's object storage

- `replit.nix`: Manages system-level dependencies
  - Specifies required system packages
  - Sets up Python 3.11
  - Configures environment variables
  - Sets up Poetry for dependency management

### Replit-Specific Features
- **Automatic Environment Detection**: The application detects Replit environment and adjusts settings
- **Optimized Defaults**: 
  - Reduced batch sizes for better performance
  - Adjusted memory usage for Replit constraints
  - Streamlined data processing pipeline
- **Integrated Storage**: Uses Replit's object storage for data persistence
- **URL Generation**: Automatically generates correct URLs for Replit environment

### Running on Replit
1. Fork the repository to your Replit account
2. Set up environment variables in Replit Secrets
3. Click the Run button or use the shell:
```bash
./scripts/replit/start.sh
```

## Docker Configuration

The project uses Docker Compose for containerized deployment with separate configurations for development and production.

### Development Setup
```bash
# Build and start development environment
docker-compose -f deployment/docker-compose.dev.yml up --build -d

# View logs
docker-compose -f deployment/docker-compose.dev.yml logs -f

# Rebuild specific service
docker-compose -f deployment/docker-compose.dev.yml up -d --build api

# Stop services
docker-compose -f deployment/docker-compose.dev.yml down
```

### Production Setup
```bash
# Build and start production environment from deployment directory
cd deployment
docker-compose -f docker-compose.yml up --build -d

# View logs
docker-compose logs -f

# Perform clean deployment (removes all data)
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Docker Environment Features
- **Multi-Service Architecture**:
  - API service (FastAPI)
  - Shared networking and volumes
- **Development Optimizations**:
  - Hot-reload for code changes
  - Volume mounting for local development
  - Debug mode enabled
  - Health checks configured
- **Production Optimizations**:
  - Non-root execution (`nobody:nogroup`)
  - Proper volume permissions
  - Multi-stage build for smaller images
  - Optimized memory and CPU allocation
  - Tini as init system for proper signal handling
  - Extended health check configuration
- **Environment Variables**:
  - Flexible configuration through .env file
  - Service-specific overrides
  - Secure secrets management
  - Enhanced model fallback configuration
  - Board-specific data filtering

Example docker-compose environment configuration:
```yaml
# Model Settings
- OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
- OPENAI_EMBEDDING_MODEL=${OPENAI_EMBEDDING_MODEL:-text-embedding-3-large}
- GROK_MODEL=${GROK_MODEL:-grok-2-1212}
- GROK_EMBEDDING_MODEL=${GROK_EMBEDDING_MODEL:-grok-v1-embedding}
- VENICE_MODEL=${VENICE_MODEL:-dolphin-2.9.2-qwen2-72b}
- VENICE_CHUNK_MODEL=${VENICE_CHUNK_MODEL:-deepseek-r1-671b}

# Data Processing Settings
- SELECT_BOARD=${SELECT_BOARD}  # Optional: Filter data by specific board
- FILTER_DATE=${FILTER_DATE}    # Optional: Filter data by date
- USE_BATCHING=${USE_BATCHING:-true}  # Control batch processing behavior
- CACHE_TTL=${CACHE_TTL:-3600}  # Cache time-to-live in seconds
- BATCH_SIZE=${BATCH_SIZE:-64}  # Size of processing batches
- AUTO_CHECK_DATA=${AUTO_CHECK_DATA:-false}  # Automatic data check on startup
```

## Testing Framework

The project includes a comprehensive testing framework to validate the Chanscope approach implementation across different environments.

### Chanscope Testing Guide

This document provides instructions for running tests to validate the Chanscope approach implementation in different environments.

#### Overview

The Chanscope approach defines a specific data processing pipeline with well-defined behaviors for different scenarios:

1. **Initial Data Load**: On application startup, data is ingested and stratified, but embedding generation is deferred.
2. **Separate Embedding Generation**: Embeddings are generated as a separate step after initial data load.
3. **Incremental Updates**: When `force_refresh=false`, existing data is used without regeneration.
4. **Forced Refresh**: When `force_refresh=true`, stratified data and embeddings are always refreshed.

#### Running Tests

##### Local Environment

To run tests in your local environment:

```bash
# Run all tests
bash scripts/run_tests.sh

# Run specific tests directly with pytest
python -m pytest tests/test_chanscope_approach.py -v
```

##### Docker Environment

To run tests in a Docker environment:

```bash
# Run all tests in Docker
cd deployment
docker-compose -f docker-compose.test.yml up --build

# Run specific test types
docker-compose -f docker-compose.test.yml up --build -e TEST_TYPE=chanscope
```

##### Replit Environment

To run tests in a Replit environment:

```bash
# Run tests with Replit-specific settings
bash scripts/replit/run_tests.sh
```

#### Integrated Test and Deploy Workflow

For a complete test-then-deploy workflow:

```bash
# Step 1: Run tests first
cd deployment
docker-compose -f docker-compose.test.yml up --build

# Step 2: If tests pass, deploy to production
docker-compose -f docker-compose.yml up -d
```

Alternatively, use the integrated test and deploy approach:

```bash
# Run tests during startup, then deploy
cd deployment
docker-compose -f docker-compose.yml up -d -e RUN_TESTS_ON_STARTUP=true
```

#### Test Results

Test results are saved to the `test_results` directory with the following files:

- `chanscope_tests_TIMESTAMP.log`: Log file with detailed test output
- `chanscope_validation_ENV_TIMESTAMP.json`: JSON file with structured test results

#### Test Descriptions

The test suite validates the following aspects of the Chanscope approach:

1. **Initial Data Load Test**: Validates that data is ingested and stratified, but embedding generation is skipped.
2. **Embedding Generation Test**: Validates that embeddings can be generated separately.
3. **force_refresh=false Test**: Validates that existing data is used without regeneration.
4. **force_refresh=true Test**: Validates that stratified data and embeddings are always refreshed.

#### Troubleshooting

If tests fail, check the following:

1. Ensure all dependencies are installed
2. Check that data directories exist and are writable
3. Verify that environment variables are set correctly
4. Check the log files for detailed error messages
5. Look for resource contention issues in shared environments
6. Check method compatibility if you see "no attribute" errors
7. Ensure proper credentials are set for S3 access

#### Environment Variables for Testing

The following environment variables affect test behavior:

- `DATA_RETENTION_DAYS`: Number of days of data to retain (default: 7)
- `ENABLE_DATA_SCHEDULER`: Whether to enable the data scheduler (default: true)
- `DATA_UPDATE_INTERVAL`: Interval in seconds between data updates (default: 1800)
- `TEST_MODE`: Set to true when running tests (default: false)
- `RUN_TESTS_ON_STARTUP`: Whether to run tests during container startup (default: false)
- `TEST_TYPE`: Type of tests to run ('all', 'chanscope', 'api', etc.)
- `ABORT_ON_TEST_FAILURE`: Whether to abort deployment if tests fail (default: true in test mode)
- `AUTO_CHECK_DATA`: Whether to automatically check and load data on startup (default: true for testing, false for production)

## Environment Variables

### Required API Keys
- `OPENAI_API_KEY`: API key for OpenAI. Required for text embeddings and completions.

### Optional API Keys
- `GROK_API_KEY`: API key for Grok (X.AI). Optional alternative provider.
- `VENICE_API_KEY`: API key for Venice.AI. Optional additional provider.

### Model Settings (Optional, with defaults)
- `OPENAI_MODEL`: OpenAI model for completions (Default: 'gpt-4o')
- `OPENAI_EMBEDDING_MODEL`: OpenAI model for embeddings (Default: 'text-embedding-3-large')
- `GROK_MODEL`: Grok model for completions (Default: 'grok-2-1212')
- `GROK_EMBEDDING_MODEL`: Grok model for embeddings (Default: 'grok-v1-embedding')
- `VENICE_MODEL`: Venice model for completions (Default: 'dolphin-2.9.2-qwen2-72b')
- `VENICE_CHUNK_MODEL`: Venice model for chunking (Default: 'deepseek-r1-671b')

### Provider Settings (Optional, with defaults)
- `DEFAULT_EMBEDDING_PROVIDER`: Default provider for embeddings (Default: 'openai')
- `DEFAULT_CHUNK_PROVIDER`: Default provider for text chunking (Default: 'openai')
- `DEFAULT_SUMMARY_PROVIDER`: Default provider for summarization (Default: 'openai')

### Processing Settings (Optional, with defaults)
- `EMBEDDING_BATCH_SIZE`: Number of items to process in each embedding batch (Default: 25)
- `CHUNK_BATCH_SIZE`: Size of chunks for processing (Default: 25000)
- `SUMMARY_BATCH_SIZE`: Number of items in each summary batch (Default: 50)
- `MAX_WORKERS`: Maximum number of parallel workers (Default: 4)
- `MAX_TOKENS`: Maximum tokens per API request (Default: 4096)
- `CACHE_ENABLED`: Enable/disable caching (Default: true)
- `CACHE_TTL`: Cache time-to-live in seconds (Default: 3600)
- `BATCH_SIZE`: Size of processing batches (Default: 64)
- `USE_BATCHING`: Enable/disable batch processing (Default: true)

### Data Processing Settings (Optional, with defaults)
- `TIME_COLUMN`: Column name for timestamps (Default: 'posted_date_time')
- `STRATA_COLUMN`: Column name for stratification (Optional)
- `FREQ`: Frequency for time-based operations (Default: 'H')
- `FILTER_DATE`: Date to filter data from (Optional)
- `SELECT_BOARD`: Filter data by specific board (Optional)
- `PADDING_ENABLED`: Enable text padding (Default: false)
- `CONTRACTION_MAPPING_ENABLED`: Enable contraction mapping (Default: false)
- `NON_ALPHA_NUMERIC_ENABLED`: Enable non-alphanumeric processing (Default: false)
- `PROCESSING_CHUNK_SIZE`: Size of chunks for processing (Default: 10000)
- `STRATIFICATION_CHUNK_SIZE`: Size of chunks for stratification (Default: 5000)

### Docker Deployment Settings (Optional, with defaults)
- `DOCKER_ENV`: Whether running in Docker environment (Default: false)
- `AUTO_CHECK_DATA`: Whether to automatically check and load data on startup (Default: false in production)
- `RUN_TESTS_ON_STARTUP`: Whether to run tests during container startup (Default: false)
- `TEST_TYPE`: Type of tests to run if RUN_TESTS_ON_STARTUP is true (Default: all)
- `ABORT_ON_TEST_FAILURE`: Whether to abort deployment if tests fail (Default: false)

### AWS Settings 
- `AWS_ACCESS_KEY_ID`: AWS access key ID (Required for S3 access)
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key (Required for S3 access)
- `AWS_DEFAULT_REGION`: AWS region (Default: 'us-east-1')
- `S3_BUCKET`: S3 bucket name (Default: 'chanscope-data')
- `S3_BUCKET_PREFIX`: S3 bucket prefix (Default: 'data/')
- `S3_BUCKET_PROCESSED`: S3 bucket for processed data (Default: 'processed')
- `S3_BUCKET_MODELS`: S3 bucket for models (Default: 'models')

(Optional, for data gathering, hourly for the last 30 days. email chanscope@proton.me for access)

### Path Settings (Optional, with defaults)
- `ROOT_DATA_PATH`: Root path for data (Default: 'data')
- `STRATIFIED_PATH`: Path for stratified data (Default: 'data/stratified')
- `PATH_TEMP`: Path for temporary files (Default: 'temp_files')

> Note: All paths are relative to the project root unless specified as absolute paths.

## Deployment Workflow

### Standard Production Deployment

```bash
# From the deployment directory:
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d
```

### Testing Before Deployment

```bash
# Run tests first
docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml up

# If tests pass, deploy to production
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d
```

### Integrated Test and Deploy

```bash
# Set RUN_TESTS_ON_STARTUP to true in production environment
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d -e RUN_TESTS_ON_STARTUP=true
```

### Clean Deployment (removes all data)

```bash
# Stop and remove all containers and volumes
docker-compose -f docker-compose.yml down -v

# Rebuild without cache
docker-compose -f docker-compose.yml build --no-cache

# Start with fresh state
docker-compose -f docker-compose.yml up -d
```

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities

### Default Configuration Settings
- **Batch Processing**:
  - Embedding Batch Size: 25
  - Chunk Batch Size: 25000
  - Summary Batch Size: 50
- **Data Processing**:
  - Stratification Chunk Size: 5000
  - Processing Chunk Size: 10000
  - Sample Size: 1500
  - Max Tokens: 4096
  - Cache Enabled: true
  - Max Workers: 4
  - Cache TTL: 3600
  - Batch Size: 64
- **Update Schedule**:
  - Hourly data refresh
  - 6-hour rolling window for real-time queries
  - 30-day data retention window