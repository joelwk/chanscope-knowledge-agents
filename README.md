# Chanscope Knowledge Agent

## Overview
An advanced query system leveraging multiple AI providers (OpenAI, Grok, Venice) for comprehensive 4chan data analysis to extract actionable insights and patterns. The system employs intelligent sampling techniques and a multi-stage analysis pipeline to process large volumes of 4chan data, enabling temporal analysis, cross-source verification, and predictive analytics.

### Platform Integration
- **Traditional Deployment**: Docker-based local or server deployment for controlled environments
- **[Replit Cloud](https://replit.com/@jwkonitzer/chanscope-knowledge-agents)**: Zero-setup cloud deployment with optimized performance
- **[Virtuals Protocol](https://app.virtuals.io/prototypes/0x2cc92Fc77180815834FfdAa72C58f72d457C4308)**: Integration with autonomous AI agents that can:
  - Process and interact with X & 4chan data through their own autonomous decision-making
  - Maintain synchronized memory across multiple data analysis sessions
  - Execute onchain transactions and data validations
  - Leverage the G.A.M.E. framework for dynamic analysis patterns
  - Enable co-owned and tokenized analysis capabilities

### Core Architecture
- **Multi-Provider Architecture**
  - OpenAI (Primary): GPT-4, text-embedding-3-large
  - Grok (Optional): grok-2-1212, grok-v1-embedding
  - Venice (Optional): llama-3.1-405b, dolphin-2.9.2-qwen2-72b

- **Intelligent Data Processing**
  - Time-based and category-based stratified sampling
  - Efficient large dataset handling with reservoir sampling
  - Automated data chunking and embedding generation
  - Cloud-optimized processing for Replit environment

- **Advanced Analysis Pipeline**
  - Context-aware temporal analysis
  - Parallel processing with automatic fallback
  - Event mapping and relationship extraction
  - Cross-platform data synchronization

The system specializes in processing 4chan data through a sophisticated pipeline that combines intelligent sampling, multi-model analysis, and cross-source verification. Through integration with Virtuals Protocol's autonomous agents, the system gains enhanced capabilities for persistent memory across analysis sessions, autonomous decision-making in data processing, and tokenized ownership of analysis models. These agents can independently interact with data sources, maintain context across multiple analyses, and execute blockchain transactions for data validation and verification.

For more detailed information on the system's technical features and example notebook, please refer to the [knowledge-agents](https://github.com/joelwk/knowledge-agents) repository.

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

## Supported Models
The project supports multiple AI model providers:
- **OpenAI** (Required): Default provider for both completions and embeddings
  - Required: `OPENAI_API_KEY`
  - Models: `gpt-4` (default), `text-embedding-3-large` (embeddings)
- **Grok (X.AI)** (Optional): Alternative provider with its own embedding model
  - Optional: `GROK_API_KEY`
  - Models: `grok-2-1212`, `grok-v1-embedding` (not publicly available)
- **Venice.AI** (Optional): Additional model provider for completion and chunking
  - Optional: `VENICE_API_KEY`
  - Models: `llama-3.1-405b`, `dolphin-2.9.2-qwen2-72b`

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

### Docker Environment Features
- **Multi-Service Architecture**:
  - API service (Python/Quart)
  - UI service (Chainlit)
  - Shared networking and volumes
- **Development Optimizations**:
  - Hot-reload for code changes
  - Volume mounting for local development
  - Debug mode enabled
  - Health checks configured
- **Environment Variables**:
  - Flexible configuration through .env file
  - Service-specific overrides
  - Secure secrets management

## API and Frontend Integration

### API Endpoints

#### 1. Query Processing
- `/process_query` (POST): Process a single query
```python
{
    "query": "Your analysis query",
    "force_refresh": false,
    "sample_size": 1500,
    "embedding_batch_size": 2000,
    "chunk_batch_size": 100,
    "summary_batch_size": 100,
    "embedding_provider": "openai",
    "chunk_provider": "openai",
    "summary_provider": "openai"
}
```

- `/batch_process` (POST): Process multiple queries in batch
```python
{
    "queries": [
        "What are the latest developments in AI?",
        "How is AI impacting healthcare?"
    ],
    "force_refresh": false,
    "sample_size": 1500,
    "embedding_batch_size": 2000,
    "chunk_batch_size": 100,
    "summary_batch_size": 100
}
```

- `/process_recent_query` (GET): Process data from the last 3 hours
```
GET /process_recent_query?force_refresh=true
```

#### 2. Health Check Endpoints
- `/health`: Basic health check
- `/health_replit`: Replit-specific health status
- `/health/connections`: Check all service connections
- `/health/s3`: Check S3 connection and bucket access
- `/health/provider/{provider}`: Check specific provider status

### Frontend Interface

The project uses Chainlit for an interactive frontend interface with the following features:

#### 1. Interactive Settings
- Model provider selection (OpenAI, Grok, Venice)
- Batch size configuration
- Data processing options
- Force refresh toggle

#### 2. Query Interface
- Real-time query processing
- Progress indicators
- Error handling and feedback
- Results display with related chunks

#### 3. Environment-Specific Features
- Automatic Replit URL detection
- Optimized settings for different environments
- Responsive UI with theme support

### Example Usage

#### 1. Using PowerShell
```powershell
# Create request body
$body = @{
    query = "What are the latest developments in AI?"
    force_refresh = $false
    sample_size = 1500
    embedding_batch_size = 2000
    chunk_batch_size = 100
    summary_batch_size = 100
    embedding_provider = "openai"
    summary_provider = "openai"
} | ConvertTo-Json

# Make request
$response = Invoke-RestMethod -Uri "http://localhost:5000/process_query" -Method Post -Body $body -ContentType "application/json"

# View results
$response.results.summary
```

#### 2. Using Bash
```bash
# Single query
curl -X POST "http://localhost:5000/process_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "force_refresh": false,
    "sample_size": 1500,
    "embedding_batch_size": 2000,
    "chunk_batch_size": 100,
    "summary_batch_size": 100,
    "embedding_provider": "openai",
    "summary_provider": "openai"
  }'

# Recent data query
curl "http://localhost:5000/process_recent_query?force_refresh=true"
```

#### 3. Using the Frontend
1. Access the UI at `http://localhost:8000`
2. Configure settings using the ⚙️ button:
   - Select model providers
   - Set batch sizes
   - Configure processing options
3. Enter your query in the chat interface
4. View results with related chunks in the sidebar

### Response Format
```python
{
    "success": true,
    "results": {
        "query": "Your original query",
        "chunks": [
            {
                "content": "Relevant text chunk",
                "score": 0.85
            }
        ],
        "summary": "Comprehensive analysis and response"
    }
}
```

## Project Structure

```
knowledge_agents/
├── api/                    # REST API service
│   ├── __init__.py        # API initialization and configuration
│   ├── app.py             # Main API application
│   └── routes.py          # API endpoint definitions
│
├── chainlit_frontend/      # Interactive UI
│   ├── app.py             # Chainlit application
│   ├── styles/            # Custom UI styles
│   └── .chainlit/         # Chainlit configuration
│
├── config/                 # Configuration files
│   ├── base.py            # Base configuration
│   ├── settings.py        # Main settings
│   ├── env_loader.py      # Environment variable loader
│   ├── logging.conf       # Logging configuration
│   └── stored_queries.yaml # Predefined queries
│
├── knowledge_agents/       # Core processing modules
│   ├── data_processing/   # Data processing components
│   │   ├── cloud_handler.py    # Cloud storage operations
│   │   ├── processing.py       # Data processing utilities
│   │   └── sampler.py          # Data sampling logic
│   ├── model_ops.py       # Model operations
│   ├── inference_ops.py   # Inference pipeline
│   ├── embedding_ops.py   # Embedding operations
│   ├── data_ops.py        # Data operations
│   ├── stratified_ops.py  # Stratified sampling
│   ├── utils.py           # Utility functions
│   └── prompt.yaml        # Model prompts
│
├── deployment/            # Deployment configurations
│   ├── Dockerfile        # Main Dockerfile
│   ├── docker-compose.yml      # Production compose
│   └── docker-compose.dev.yml  # Development compose
│
├── scripts/              # Utility scripts
│   └── replit/          # Replit-specific scripts
│       ├── api.sh       # API startup script
│       └── ui.sh        # UI startup script
│
├── tests/               # Test suite
│   ├── test_endpoints.py     # API endpoint tests
│   └── debug_client.py       # Debug utilities
│
├── examples/            # Example notebooks and scripts
├── data/               # Data directory
├── logs/               # Log files
├── temp_files/         # Temporary processing files
│
├── .env.template       # Environment template
├── .replit            # Replit configuration
├── replit.nix         # Replit Nix configuration
├── pyproject.toml     # Poetry dependencies
├── poetry.lock        # Poetry lock file
└── README.md          # Project documentation
```

### Key Components

#### 1. Core Processing (`knowledge_agents/`)
- Data processing pipeline and utilities
- Model operations and inference
- Embedding and stratification logic
- Utility functions and configurations

#### 2. API Service (`api/`)
- REST API implementation
- Route definitions
- Request handling
- Health checks and monitoring

#### 3. Frontend (`chainlit_frontend/`)
- Interactive UI implementation
- Custom styling
- User session management
- Real-time updates

#### 4. Configuration (`config/`)
- Environment management
- Application settings
- Logging configuration
- Query templates

#### 5. Deployment (`deployment/`)
- Docker configurations
- Environment-specific setups
- Service orchestration

#### 6. Development Tools
- Testing framework
- Debug utilities
- Example implementations
- Documentation

#### 7. Replit Integration
- Replit-specific configurations
- Startup scripts
- Environment optimizations
- Cloud storage integration

## Development

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
poetry install
```

### Development Workflow

#### 1. Local Development
```bash
# Start API server in development mode
python -m api.app --debug

# Start Chainlit frontend in development mode
python -m chainlit_frontend.app --debug
```

#### 2. Docker Development
```bash
# Build and start development environment
docker-compose -f deployment/docker-compose.dev.yml up --build

# View logs
docker-compose -f deployment/docker-compose.dev.yml logs -f

# Rebuild and restart specific service
docker-compose -f deployment/docker-compose.dev.yml up -d --build api

# Stop services
docker-compose -f deployment/docker-compose.dev.yml down
```

### Testing Workflow

#### 1. Local Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_endpoints.py

# Run tests with coverage
pytest --cov=knowledge_agents tests/

# Run tests with verbose output
pytest -v tests/
```

#### 2. Docker Testing
```bash
# Run tests using docker-compose
docker-compose -f deployment/docker-compose.dev.yml -f deployment/Dockerfile.test up --build test

# View test logs
docker-compose -f deployment/docker-compose.dev.yml logs test
```

### Service Management

#### Health Checks and Monitoring
```bash
# Check API health
curl http://localhost:5000/health

# Check Chainlit frontend
Open http://localhost:8000 in your browser

# View service status
docker-compose -f deployment/docker-compose.dev.yml ps

# Monitor logs
docker-compose -f deployment/docker-compose.dev.yml logs -f
```

## Environment Variables

### Required API Keys
- `OPENAI_API_KEY`: API key for OpenAI. Required for text embeddings and completions.

### Optional API Keys
- `GROK_API_KEY`: API key for Grok (X.AI). Optional alternative provider.
- `VENICE_API_KEY`: API key for Venice.AI. Optional additional provider.

### Model Settings (Optional, with defaults)
- `OPENAI_MODEL`: OpenAI model for completions (Default: 'gpt-4')
- `OPENAI_EMBEDDING_MODEL`: OpenAI model for embeddings (Default: 'text-embedding-3-large')
- `GROK_MODEL`: Grok model for completions (Default: 'grok-2-1212')
- `GROK_EMBEDDING_MODEL`: Grok model for embeddings (Default: 'grok-v1-embedding')
- `VENICE_MODEL`: Venice model for completions (Default: 'llama-3.1-405b')
- `VENICE_CHUNK_MODEL`: Venice model for chunking (Default: 'dolphin-2.9.2-qwen2-72b')

### Provider Settings (Optional, with defaults)
- `DEFAULT_EMBEDDING_PROVIDER`: Default provider for embeddings (Default: 'openai')
- `DEFAULT_CHUNK_PROVIDER`: Default provider for text chunking (Default: 'openai')
- `DEFAULT_SUMMARY_PROVIDER`: Default provider for summarization (Default: 'openai')

### Processing Settings (Optional, with defaults)
- `BATCH_SIZE`: Number of items to process in each batch (Default: 100)
- `MAX_WORKERS`: Maximum number of parallel workers (Default: 4)
- `MAX_TOKENS`: Maximum tokens per API request (Default: 2048)
- `CHUNK_SIZE`: Size of text chunks for processing (Default: 1000)
- `CACHE_ENABLED`: Enable/disable caching (Default: true)

### Data Processing Settings (Optional, with defaults)
- `TIME_COLUMN`: Column name for timestamps (Default: 'posted_date_time')
- `STRATA_COLUMN`: Column name for stratification (Optional)
- `FREQ`: Frequency for time-based operations (Default: 'H')
- `FILTER_DATE`: Date to filter data from (Optional)
- `PADDING_ENABLED`: Enable text padding (Default: false)
- `CONTRACTION_MAPPING_ENABLED`: Enable contraction mapping (Default: false)
- `NON_ALPHA_NUMERIC_ENABLED`: Enable non-alphanumeric processing (Default: false)

### AWS Settings 
- `AWS_ACCESS_KEY_ID`: AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key
- `AWS_DEFAULT_REGION`: AWS region (Default: 'us-east-1')
- `S3_BUCKET`: S3 bucket name (Default: 'rolling-data')
- `S3_BUCKET_PREFIX`: S3 bucket prefix (Default: 'data')

(Optional, for data gathering, hourly for the last 30 days. email chanscope@proton.me for access)

### Path Settings (Optional, with defaults)
- `ROOT_PATH`: Root path for data (Default: 'data')
- `DATA_PATH`: Path for data files (Default: 'data')
- `ALL_DATA`: Path for all data file (Default: 'data/all_data.csv')
- `ALL_DATA_STRATIFIED_PATH`: Path for stratified data (Default: 'data/stratified')
- `KNOWLEDGE_BASE`: Path for knowledge base (Default: 'data/knowledge_base.csv')
- `PATH_TEMP`: Path for temporary files (Default: 'temp_files')

> Note: All paths are relative to the project root unless specified as absolute paths.
## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities