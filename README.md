# Chanscope Retrieval

## Overview
An advanced query system leveraging multiple AI providers (OpenAI, Grok, Venice) for comprehensive social data analysis to extract actionable insights and patterns. The system provides a robust API layer that can be integrated with autonomous AI agents and agentic systems. It employs intelligent sampling techniques and a multi-stage analysis pipeline to process large volumes of 4chan and X data, enabling temporal analysis, cross-source verification, and predictive analytics.

### Platform Integration
- **Traditional Deployment**: Docker-based local or server deployment for controlled environments
- **[Replit Cloud](https://replit.com/@jwkonitzer/chanscope-knowledge-agents)**: 
  - Zero-setup cloud deployment with optimized performance
  - Automated hourly data updates
  - Environment-specific path handling
  - Optimized memory management

### Current Uses 
- **Agentic System Integration**: 
  - **[Virtuals Protocol](https://app.virtuals.io/prototypes/0x2cc92Fc77180815834FfdAa72C58f72d457C4308)/[X](https://x.com/4Chanscope)**: The system can be integrated with autonomous AI agents through its API layer, enabling:
    - Consumption of 4chan and X data analysis through standardized API endpoints
    - Integration with agent memory systems for persistent context
    - Support for agent-driven data exploration and pattern recognition
    - Potential for onchain data validation and verification
    - Extensibility for custom agent-specific analysis patterns

## System Architecture

Chanscope's architecture follows a biologically-inspired pattern with distinct yet interconnected processing stages:

```
┌─────────────────┐         ┌──────────────────────────┐         ┌─────────────────┐
│   Data Sources  │         │    Processing Core       │         │  Query System   │
│  ┌────────────┐ │         │  ┌────────────────────┐  │         │ ┌────────────┐  │
│  │    S3      │◄├─┐       │  │ ChanScopeDataMgr   │  │     ┌───┼►│   Query    │  │
│  │  Storage   │ │ │       │  │ ┌────────────────┐ │  │     │   │ │ Processing │  │
│  └────────────┘ │ │       │  │ │   Stratified   │ │  │     │   │ └─────┬──────┘  │
└─────────────────┘ │       │  │ │    Sampling    │ │  │     │   │       │         │
                    │       │  │ └────────┬───────┘ │  │     │   │ ┌─────▼──────┐  │
┌─────────────────┐ │       │  │          │         │  │     │   │ │   Chunk    │  │
│  Memory System  │ │       │  │ ┌────────▼───────┐ │  │     │   │ │ Processing │  │
│  ┌────────────┐ │ │       │  │ │   Embedding    │ │  │     │   │ └─────┬──────┘  │
│  │ Complete   │◄┼─┘       │  │ │   Generation   │ │  │     │   │       │         │
│  │    Data    │ │         │  │ └────────────────┘ │  │     │   │ ┌─────▼──────┐  │
│  └────────────┘ │         │  └────────────────────┘  │     │   │ │   Final    │  │
│  ┌────────────┐ │         │           │              │     │   │ │ Summarizer │  │
│  │ Stratified │◄├─────────┼───────────┘              │     │   │ └────────────┘  │
│  │   Sample   │ │         │                          │     │   │                 │
│  └────────────┘ │         │  ┌────────────────────┐  │     │   └─────────────────┘
│  ┌────────────┐ │         │  │   KnowledgeAgent   │  │     │
│  │ Embeddings │◄├─────────┼──┤  (Singleton LLM)   ├──┼─────┘
│  │   (.npz)   │ │         │  └────────────────────┘  │
│  └────────────┘ │         └──────────────────────────┘
└─────────────────┘
           ▲
           │
    ┌──────┴──────┐
    │ Storage ABCs │
    └─────────────┘
```

### Processing Pipeline

1. **Data Ingestion**: `ChanScopeDataManager` retrieves data from S3 starting from `DATA_RETENTION_DAYS` ago, using the appropriate storage implementation.
2. **Stratification**: Samples the complete dataset using `sampler.py` to create a representative subset, with file-locks for concurrent access management.
3. **Embedding Generation**: Creates embeddings via `KnowledgeAgent` singleton and stores them in environment-specific format (`.npz` files or Object Storage) with thread ID mappings.
4. **Query Processing**: Performs vector similarity search using cosine distance and incorporates an enhanced natural language query processing module. Uses batch processing for efficiency and supports recursive refinement for improved results.

## Repository Structure

```
├── api/                  # FastAPI application and endpoints
│   ├── app.py            # Main API application with lifespan management
│   ├── routes.py         # API route definitions
│   ├── models.py         # Data models and schemas
│   ├── cache.py          # Caching mechanisms
│   └── errors.py         # Error handling
├── config/               # Configuration files and settings
│   ├── storage.py        # Storage abstraction interfaces & implementations
│   ├── settings.py       # Configuration management
│   ├── env_loader.py     # Environment detection
│   └── chanscope_config.py # Chanscope-specific configuration
├── deployment/           # Docker and deployment configurations
├── docs/                 # Documentation files
├── knowledge_agents/     # Core business logic and data processing
│   ├── data_ops.py       # Data operations and processing
│   ├── embedding_ops.py  # Embedding generation and management
│   ├── inference_ops.py  # Inference and query processing
│   ├── model_ops.py      # Model management and LLM operations
│   ├── llm_sql_generator.py # Natural language to SQL conversion
│   ├── prompt.yaml       # LLM prompt templates
│   ├── data_processing/  # Data processing subpackage
│   │   ├── chanscope_manager.py # Central facade for data operations
│   │   ├── cloud_handler.py # S3/GCS abstraction
│   │   ├── sampler.py    # Stratified sampling implementation
│   │   └── dialog_processor.py # Text processing utilities
│   └── run.py            # Main execution logic
├── scripts/              # Utility scripts for testing and deployment
├── tests/                # Test suites and fixtures
└── examples/             # Example usage and integrations
```

## Core Architecture

- **Multi-Provider Architecture**
  - Singleton `KnowledgeAgent` provides unified access to different LLM providers
  - OpenAI (Primary): GPT-4o, text-embedding-3-large
  - Grok (Optional): grok-3, grok-3-mini
  - Venice (Optional): dolphin-2.9.2-qwen2-72b, deepseek-r1-671b

- **Storage Abstraction Layer**
  - Abstract interfaces: `CompleteDataStorage`, `StratifiedSampleStorage`, `EmbeddingStorage`, `StateManager`
  - `StorageFactory` selects appropriate implementation based on environment
  - File-based implementations for Docker/local environments
  - Replit implementations using PostgreSQL, Key-Value store, and Object Storage
  - Thread-safe operations with file locks for concurrent access

- **Intelligent Data Processing**
  - Automated hourly data updates with incremental processing
  - Time-based and category-based stratified sampling with configurable weights
  - Board-specific data filtering and validation
  - Efficient large dataset handling with reservoir sampling
  - Automated data chunking and embedding generation
  - Configurable data retention with `DATA_RETENTION_DAYS` environment variable
  - Robust process locking to prevent duplicate processing:
    - Uses Replit Object Storage for persistent locks in Replit environments
    - Uses file-based locks in Docker/local environments
    - Tracks initialization status to avoid redundant processing
  - Three-stage data processing pipeline:
    1. Complete data ingestion and storage
    2. Stratified sample generation
    3. Embedding generation and storage
  - Flexible regeneration options:
    - `--regenerate --stratified-only`: Regenerate only stratified sample
    - `--regenerate --embeddings-only`: Regenerate only embeddings
    - `--force-refresh`: Force refresh all data stages
    - `--ignore-lock`: Bypass process locks (use with caution)
  - Environment-specific storage backends:
    - Replit: PostgreSQL for complete data, Key-Value store for stratified samples, Object Storage for embeddings
    - Docker: File-based storage with CSV, NPZ, and JSON formats

- **Advanced Analysis Pipeline**
  - Real-time monitoring with 6-hour rolling window
  - Context-aware temporal analysis with validation
  - Parallel processing with automatic model fallback
  - Event mapping and relationship extraction
  - Cross-platform data synchronization
  - Enhanced S3 data streaming with board filtering
  - Optimized batch processing for query efficiency
  - Dual processing modes with `force_refresh` flag:
    - When enabled: Regenerates stratified samples and embeddings
    - When disabled: Uses existing data for faster processing

- **LLM-Based SQL Generation**
  - Hybrid approach combining template matching and LLM generation
  - Three-stage LLM pipeline:
    1. **Enhancer**: Refines natural language query into structured instructions
    2. **Generator**: Converts enhanced instructions to SQL (uses Venice characters)
    3. **Validator**: Ensures security and correctness of generated SQL
  - Template matching for common query patterns with fallback to LLM
  - Parameter extraction with time-awareness
  - Full schema validation and security checks
  - Caching for improved performance

- **API-First Design**
  - RESTful endpoints for all core functionality
  - Structured JSON responses for easy integration
  - Comprehensive error handling with detailed feedback
  - Batch processing for high-volume requests
  - Authentication and rate limiting for production use
  - Persistent task tracking with detailed status reporting
  - Automatic cleanup of old results with history preservation
  - Background processing with `use_background` parameter
  - Custom task IDs for integration with external systems

## Component Relationships

- **ChanScopeDataManager**: Central facade that orchestrates all data operations through environment-specific storage interfaces
- **KnowledgeAgent**: Singleton service providing unified access to LLM providers for embeddings, chunking, and summarization
- **Storage ABCs**: Abstract interfaces allowing seamless switching between file-based and database storage
- **Model and Embedding Operations**: Separate modules that handle model interactions and embedding management
- **API Layer**: FastAPI application that initializes ChanScopeDataManager once and exposes its functionality through routes
- **LLMSQLGenerator**: Specialized component that converts natural language to SQL using a hybrid template/LLM approach

For greater technical details and examples, refer to the documentation in the `docs/` directory and the [knowledge-agents](https://github.com/joelwk/knowledge-agents) repository.

## Analysis Capabilities

### 1. Temporal Analysis
- Thread dynamics tracking
- Activity burst detection
- Topic evolution mapping
- Cross-reference analysis
- Real-time trend prediction

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

The Chanscope Retrieval is designed to serve as a backend for AI agents and agentic systems through its API layer:

### Agent Integration Patterns
- **Direct API Consumption**: Agents can directly query the API endpoints
- **Memory Augmentation**: Results can be stored in agent memory systems
- **Decision Support**: Analysis can inform agent decision-making processes
- **Autonomous Monitoring**: Agents can set up scheduled queries for monitoring

### Agent Capabilities Enabled
- **Contextual Understanding**: Deep understanding of 4chan discussions and trends
- **Pattern Recognition**: Identification of emerging patterns and anomalies
- **Temporal Awareness**: Understanding of how topics evolve over time
- **Cross-Reference Analysis**: Connecting related discussions across threads and boards

## Environment Configuration

The project uses an intelligent environment detection system that automatically configures settings based on the deployment context:

### Environment Detection
- **Replit Detection**: Automatically detects Replit environment through multiple indicators
- **Docker Detection**: Identifies Docker containers through environment markers
- **Local Development**: Falls back to local configuration when neither is detected

### Environment-Specific Settings
1. **Replit Environment**:
   - Optimized path configuration for Replit filesystem
   - Automatic directory structure creation
   - Memory-optimized batch sizes and worker counts
   - Default configuration for mock data and embeddings
   - Process lock management using Object Storage to prevent duplicate processing
   - Environment-aware initialization that handles development vs deployment differences

2. **Docker Environment**:
   - Container-specific path configuration
   - Optimized worker counts for containerized deployment
   - Enhanced batch processing settings
   - Automatic volume management
   - File-based process lock management

3. **Local Environment**:
   - Flexible path configuration
   - Development-friendly defaults
   - Easy-to-modify settings
   - File-based process lock management

### Configuration Sections
The `.env` file supports section-based configuration:
```ini
[replit]
# Replit-specific settings
USE_MOCK_DATA=false
EMBEDDING_BATCH_SIZE=10
MAX_WORKERS=2

[docker]
# Docker-specific settings
EMBEDDING_BATCH_SIZE=20
MAX_WORKERS=4

[local]
# Local development settings
```

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/joelwk/knowledge-agents.git
cd knowledge-agents
cp .env.template .env  # Configure your API keys
```

### 2. Required Environment Variables
- `OPENAI_API_KEY`: Primary provider (Required)
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: For S3 access (Required)
- `DATA_RETENTION_DAYS`: Number of days to retain data (Optional, defaults to 30)

### 3. Environment-Specific Configuration
The system automatically detects and configures based on your environment:

#### Replit Deployment
```bash
# Set in Replit Secrets:
OPENAI_API_KEY=your_key
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_key
S3_BUCKET=your_bucket
```

#### Docker Deployment
```bash
# In your .env file:
DOCKER_ENV=true
EMBEDDING_BATCH_SIZE=20
MAX_WORKERS=4
```

#### Local Development
```bash
# In your .env file:
# Leave DOCKER_ENV and REPLIT_ENV unset for local detection
```

### 4. Launch Application
```bash
# For Docker
docker-compose -f deployment/docker-compose.yml up --build -d

# Access Services
API: http://localhost:80
```

### 5. Basic API Usage

#### Synchronous Query
```bash
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy",
    "force_refresh": false
  }'
```

#### Background Processing
```bash
# Submit background task
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bitcoin Strategic Reserve",
    "use_background": true,
    "task_id": "bitcoin_analysis_123"
  }'

# Check task status
curl -X GET "http://localhost/api/v1/batch_status/bitcoin_analysis_123"
```

#### Natural Language Database Query
```bash
# Query using natural language
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts about Bitcoin from last week",
    "limit": 20
  }'
```

## Deployment Options

The project supports multiple deployment options:

### Docker Deployment
For detailed Docker deployment instructions, see [deployment/README_DEPLOYMENT.md](deployment/README_DEPLOYMENT.md)

### Replit Deployment
The project is configured to run seamlessly on Replit with optimized settings:

1. Fork the repository to your Replit account
2. Set up environment variables in Replit Secrets:
   ```
   OPENAI_API_KEY=your_key
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_key
   S3_BUCKET=your_bucket
   ```
3. Click the Run button to start the application
4. The system will automatically detect the Replit environment and:
   - Install required dependencies including replit-object-storage
   - Initialize the PostgreSQL schema
   - Use Object Storage for process locks and initialization status
   - Prevent duplicate data processing during restarts
   - Run data processing in the background
   - Perform hourly data updates if enabled

## Testing Framework

The project includes a comprehensive testing framework to validate functionality across different environments:

- **Data Ingestion Tests**: Validate S3 data retrieval and processing
- **Embedding Tests**: Validate embedding generation and storage
- **API Endpoint Tests**: Validate API functionality
- **Chanscope Approach Tests**: Validate the complete pipeline
- **Task Management Tests**: Verify background processing and status tracking
- **Force Refresh Tests**: Ensure proper behavior with different refresh settings

For detailed testing instructions, see [tests/README_TESTING.md](tests/README_TESTING.md)

## API Endpoints

The Chanscope Retrieval provides a comprehensive set of API endpoints for querying and managing data:

- **Health Check Endpoints**: Various health check endpoints to verify system status
- **Query Processing Endpoints**: Synchronous and asynchronous query processing
- **Batch Processing**: Process multiple queries in a batch
- **Data Management**: Endpoints for triggering data stratification and embedding generation
- **Task Management**: Enhanced task status tracking with persistent history
- **Natural Language Queries**: Convert natural language to SQL for database queries

For detailed API usage examples, see [api/README_REQUESTS.md](api/README_REQUESTS.md)

## Supported Models
The project supports multiple AI model providers:
- **OpenAI** (Required): Default provider for both completions and embeddings
- **Grok (X.AI)** (Optional): Alternative provider for completions and chunking
- **Venice.AI** (Optional): Additional model provider for completion and chunking

## Documentation

- **[deployment/README_DEPLOYMENT.md](deployment/README_DEPLOYMENT.md)**: Detailed deployment instructions
- **[tests/README_TESTING.md](tests/README_TESTING.md)**: Comprehensive testing framework documentation
- **[docs/chanscope_implementation.md](docs/chanscope_implementation.md)**: Implementation details and technical specifications
- **[docs/llm_sql_feature.md](docs/llm_sql_feature.md)**: LLM-based SQL generation documentation
- **[docs/stratification_guide.md](docs/stratification_guide.md)**: Stratification best practices
- **[api/README_REQUESTS.md](api/README_REQUESTS.md)**: API usage examples

## Environment Variables
For a complete and up-to-date list of environment variables, see [.env.template](.env.template)

### Data Processing Control Variables
- `AUTO_CHECK_DATA`: Enable/disable automatic data checking on startup (defaults to true)
- `CHECK_EXISTING_DATA`: Check if data already exists in database before processing (defaults to true)
- `FORCE_DATA_REFRESH`: Force refresh data even if fresh data exists (defaults to false)
- `SKIP_EMBEDDINGS`: Skip embedding generation during data processing (defaults to false)
- `DATA_RETENTION_DAYS`: Number of days to retain data (defaults to 14)
- `DATA_UPDATE_INTERVAL`: How often to update data in seconds (defaults to 86400, once per day)

## Test Data Generation

For testing purposes when real data is unavailable or outdated, you can generate synthetic test data:

```bash
# Generate 1000 rows of synthetic data with timestamps in the past 10 days
poetry run python scripts/generate_test_data.py

# Generate 5000 rows with specific date range and regenerate stratified sample & embeddings
poetry run python scripts/generate_test_data.py --num-rows 5000 --start-date 2025-03-01T00:00:00 --end-date 2025-03-30T23:59:59 --regenerate-stratified --regenerate-embeddings
```

You can also adjust the `FILTER_DATE` environment variable to include older test data:

```bash
# Set a specific filter date in .env or environment
export FILTER_DATE=2024-04-01  # Include data from April 2024 onwards
```

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities

### Data Processing Commands

Basic data processing:
```bash
# Process all data stages
poetry run python scripts/process_data.py

# Check current data status (includes initialization status)
poetry run python scripts/process_data.py --check

# Force refresh all data
poetry run python scripts/process_data.py --force-refresh

# Regenerate specific components
poetry run python scripts/process_data.py --regenerate --stratified-only  # Only regenerate stratified sample
poetry run python scripts/process_data.py --regenerate --embeddings-only  # Only regenerate embeddings

# Advanced options
poetry run python scripts/process_data.py --ignore-lock  # Bypass process locks (use with caution)
```

### Process Lock Management

The system includes a robust process lock management mechanism to prevent duplicate data processing:

```bash
# Test process lock functionality
poetry run python scripts/test_process_lock.py --all

# Test specific lock features
poetry run python scripts/test_process_lock.py --test-contention  # Test lock contention between processes
poetry run python scripts/test_process_lock.py --test-marker  # Test initialization markers
```

In Replit environments, the lock manager uses Object Storage for persistence across restarts, while in Docker/local environments it uses file-based locks. This ensures that:

1. Development mode in Replit won't start redundant data processing on restarts
2. Deployment mode in Replit will have proper process initialization through FastAPI lifecycle
3. Docker and local environments have appropriate lock management for their contexts