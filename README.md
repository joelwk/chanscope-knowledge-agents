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
  - Time-based and category-based stratified sampling
  - Board-specific data filtering and validation
  - Efficient large dataset handling with reservoir sampling
  - Automated data chunking and embedding generation

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
  
  For AWS S3 access to 4chan data (hourly for the last 30 days), email chanscope@proton.me.

### 3. Launch with Docker
```bash
# Development
docker-compose -f deployment/docker-compose.yml up --build -d

# Access Services
API: http://localhost:80
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

For detailed testing instructions, see [tests/README_TESTING.md](tests/README_TESTING.md)

## API Endpoints

The Knowledge Agent provides a comprehensive set of API endpoints for querying and managing data:

- **Health Check Endpoints**: Various health check endpoints to verify system status
- **Query Processing Endpoints**: Synchronous and asynchronous query processing
- **Batch Processing**: Process multiple queries in a batch
- **Data Management**: Endpoints for triggering data stratification and embedding generation

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

For a complete list of environment variables and their descriptions, see [.env.template](.env.template)

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities