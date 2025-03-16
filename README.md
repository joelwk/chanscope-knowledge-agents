# Chanscope Retrieval

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

## Repository Structure

```
├── api/                  # FastAPI application and endpoints
│   ├── app.py            # Main API application
│   ├── routes.py         # API route definitions
│   ├── models.py         # Data models and schemas
│   ├── cache.py          # Caching mechanisms
│   └── errors.py         # Error handling
├── config/               # Configuration files and settings
├── deployment/           # Docker and deployment configurations
│   ├── docker-compose.yml       # Production deployment configuration
│   ├── docker-compose.test.yml  # Testing deployment configuration
│   ├── Dockerfile               # Container definition
│   └── setup.sh                 # Setup script for container initialization
├── docs/                 # Documentation files
│   └── chanscope_implementation.md  # Implementation details
├── knowledge_agents/     # Core business logic and data processing
│   ├── data_ops.py       # Data operations and processing
│   ├── embedding_ops.py  # Embedding generation and management
│   ├── inference_ops.py  # Inference and query processing
│   ├── model_ops.py      # Model management and configuration
│   └── run.py            # Main execution logic
├── scripts/              # Utility scripts for testing and deployment
│   ├── run_tests.sh      # Main test runner
│   ├── docker_tests.sh   # Docker-specific testing
│   ├── local_tests.sh    # Local environment testing
│   ├── replit_tests.sh   # Replit environment testing
│   └── test_and_deploy.sh  # Combined test and deployment workflow
├── tests/                # Test suites and fixtures
│   ├── test_data_ingestion.py     # Data ingestion tests
│   ├── test_embedding_pipeline.py # Embedding generation tests
│   ├── test_endpoints.py          # API endpoint tests
│   └── test_chanscope_approach.py # Chanscope approach tests
└── examples/             # Example usage and integrations
```

## Core Architecture

- **Multi-Provider Architecture**
  - OpenAI (Primary): GPT-4o, text-embedding-3-large
  - Grok (Optional): grok-2-1212, grok-v1-embedding
  - Venice (Optional): dolphin-2.9.2-qwen2-72b, deepseek-r1-671b

- **Intelligent Data Processing**
  - Automated hourly data updates with incremental processing
  - Time-based and category-based stratified sampling with configurable weights
  - Board-specific data filtering and validation
  - Efficient large dataset handling with reservoir sampling
  - Automated data chunking and embedding generation
  - Configurable data retention with `DATA_RETENTION_DAYS` environment variable

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

For greater technical details and examples, refer to the [knowledge-agents](https://github.com/joelwk/knowledge-agents) repository.

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

2. **Docker Environment**:
   - Container-specific path configuration
   - Optimized worker counts for containerized deployment
   - Enhanced batch processing settings
   - Automatic volume management

3. **Local Environment**:
   - Flexible path configuration
   - Development-friendly defaults
   - Easy-to-modify settings

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

## Deployment Options

The project supports multiple deployment options:

### Docker Deployment
For detailed Docker deployment instructions, see [deployment/README_DEPLOYMENT.md](deployment/README_DEPLOYMENT.md)

### Replit Deployment
The project is configured to run seamlessly on Replit with optimized settings:

1. Fork the repository to your Replit account
2. Set up environment variables in Replit Secrets
3. Click the Run button or use the appropriate startup script for Replit

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

For detailed API usage examples, see [tests/knowledge_agent_api_tests.md](tests/knowledge_agent_api_tests.md)

## Supported Models
The project supports multiple AI model providers:
- **OpenAI** (Required): Default provider for both completions and embeddings
  - Models: `gpt-4o` (default), `text-embedding-3-large` (embeddings)
- **Grok (X.AI)** (Optional): Alternative provider with its own embedding model
  - Models: `grok-2-1212`, `grok-v1-embedding` (not publicly available)
- **Venice.AI** (Optional): Additional model provider for completion and chunking
  - Models: `dolphin-2.9.2-qwen2-72b`, `deepseek-r1-671b`

## Documentation

- **[deployment/README_DEPLOYMENT.md](deployment/README_DEPLOYMENT.md)**: Detailed deployment instructions
- **[scripts/README_TESTING.md](scripts/README_TESTING.md)**: Comprehensive testing framework documentation
- **[docs/chanscope_implementation.md](docs/chanscope_implementation.md)**: Implementation details and technical specifications
- **[tests/knowledge_agent_api_tests.md](tests/knowledge_agent_api_tests.md)**: API usage examples

## Environment Variables
For a complete and up-to-date list of environment variables, see [.env.template](.env.template)

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities