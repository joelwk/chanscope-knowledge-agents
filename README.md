# Chanscope Knowledge Agent

## Description
An advanced query system/tool that leverages multiple AI providers (OpenAI, Grok, Venice) to analyze 4chan data (can be extended to other sources). The system employs a multi-stage pipeline with advanced sampling, prompt engineering, and inference capabilities to generate insights, with a focus on temporal analysis, event mapping and forecasting. 

1. **Data Preparation & Sampling**: 
   - Intelligent data filtering and stratified sampling
   - Time-based and category-based data segmentation
   - Reservoir sampling for large dataset handling

2. **Embedding Generation**: 
   - Creates semantic embeddings for efficient text search and retrieval
   - Multi-provider support with automatic fallback
   - Batch processing for optimal performance

3. **Inference & Analysis**:
   - Agent-based text chunking and parallel processing
   - Context-aware summarization with temporal analysis
   - Event mapping and relationship extraction

## Core Components

### Data Processing & Sampling (sampler.py)
The Sampler module is a cornerstone of the system, providing sophisticated data filtering and sampling capabilities:
- Time-based filtering with configurable thresholds
- Stratified sampling across multiple dimensions
- Hybrid sampling strategies (time + strata)
- Reservoir sampling for handling large datasets

### Inference Pipeline (inference_ops.py)
The Inference Operations module orchestrates the core analysis workflow:
- Intelligent text chunking with token/character limit awareness
- Parallel summarization of content chunks
- Context preservation across summaries
- Optional temporal and event relationship mapping

Key features:
- Multiple model provider support with automatic fallback mechanisms
- Temporal-aware analysis for tracking information evolution over time
- Batch processing capabilities for multiple queries
- REST API endpoints for easy integration
- Docker containerization for simplified deployment
- Configurable model selection for each pipeline stage

The system is designed to process text data through specialized models, each optimized for their specific task. This modular approach allows for flexible model selection while maintaining robust operation through automatic fallback mechanisms.

For more detailed information on the system's technical features, please refer to the [knowledge-agents](https://github.com/joelwk/knowledge-agents) repository.
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

## Setup Instructions

### 1. Environment Configuration
1. Clone the repository:
```bash
git clone https://github.com/joelwk/knowledge-agents.git
cd knowledge-agents
```

2. Set up environment variables:
```bash
# Copy the template file
cp .env.template .env

# Edit .env with your API keys and settings
nano .env  # or use your preferred editor
```

Required configurations in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-large)

Optional configurations:
- Grok API settings (if using Grok)
- Venice API settings (if using Venice)
- AWS credentials (if using data gathering features)

### 2. Docker Setup

#### Development Environment
1. Build and start the development services:
```bash
# Start the development environment
docker-compose -f deployment/docker-compose.dev.yml up --build -d

# View logs
docker-compose -f deployment/docker-compose.dev.yml logs -f
```

2. The development setup includes:
   - API service on port 5000
   - Chainlit frontend on port 8000
   - Hot-reloading enabled for both services

#### Testing Environment
To run the test suite:
```bash
# Build and run tests
docker-compose -f deployment/docker-compose.dev.yml -f deployment/Dockerfile.test up --build test
```

3. Verify the services are running:
```bash
# Check API health
curl http://localhost:5000/health

# Access Chainlit frontend
Open http://localhost:8000 in your browser
```

## Making Requests

### Using PowerShell:
```powershell
# Single query
$body = @{
    query = "What are the latest developments in AI?"
    process_new = $true
    batch_size = 100
    embedding_provider = "openai"
    summary_provider = "venice"
} | ConvertTo-Json

# Make request and display full response
$response = Invoke-RestMethod -Uri "http://localhost:5000/process_query" -Method Post -Body $body -ContentType "application/json"
$response | ConvertTo-Json -Depth 10

# Display just the summary
$response.results.summary
```

### Using curl (bash/cmd):
```bash
# Single query
curl -X POST "http://localhost:5000/process_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "process_new": true,
    "batch_size": 100,
    "embedding_provider": "openai",
    "summary_provider": "venice"
  }'

# Batch processing
curl -X POST "http://localhost:5000/batch_process" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What are the latest developments in AI?",
      "How is AI impacting healthcare?"
    ],
    "process_new": false,
    "batch_size": 100,
    "embedding_provider": "openai",
    "summary_provider": "venice"
  }'
```

### API Parameters
- `query`: The search query (required)
- `process_new`: Whether to process new data (default: false)
- `batch_size`: Number of items to process (default: 100)
- `embedding_provider`: Provider for embeddings ("openai" or "grok")
- `summary_provider`: Provider for summaries ("openai", "grok", or "venice")

## Data Gathering
For data collection functionality, you can utilize the data gathering tools from the [chanscope-lambda repository](https://github.com/joelwk/chanscope-lambda). If you prefer not to set up a Lambda function, you can use the `gather.py` script directly from that repository for data collection purposes.

### Using gather.py
1. Clone the chanscope-lambda repository
2. Navigate to the gather.py script
3. Follow the script's documentation for standalone data gathering functionality

## Project Structure
```
knowledge_agents/
├── /knowledge_agents/           # Core knowledge processing functionality
│   ├── /data_processing/       # Data processing and sampling modules
│   │   ├── sampler.py         # Core sampling and filtering logic
│   │   ├── cloud_handler.py   # S3 data management
│   │   ├── processing.py      # Text preprocessing utilities
│   │   └── contraction_mapping.json  # Text normalization rules
│   ├── model_ops.py           # Model provider management
│   ├── data_ops.py           # Data orchestration and preparation
│   ├── inference_ops.py      # Core inference and analysis pipeline
│   ├── embedding_ops.py      # Embedding generation and management
│   ├── stratified_ops.py     # Data stratification utilities
│   ├── utils.py             # Common utilities and helpers
│   ├── prompt.yaml          # Analysis prompt configurations
│   └── run.py              # Pipeline orchestration
│
├── /api/                    # REST API Layer
│   ├── __init__.py         # Package initialization
│   ├── app.py              # Flask application setup
│   └── routes.py           # API endpoint definitions
│
├── /chainlit_frontend/      # Interactive Analysis Frontend
│   ├── app.py              # Chainlit application
│   ├── chainlit.yaml       # Chainlit configuration
│   ├── __init__.py         # Frontend initialization
│   └── /styles             # UI styling
│       └── custom.css      # Custom CSS styles
│
├── /config/                # Configuration
│   ├── settings.py        # Application settings
│   └── logging.conf       # Logging configuration
│
├── /deployment/           # Deployment Configuration
│   ├── docker-compose.yml # Container orchestration
│   ├── Dockerfile        # Container build configuration
│   └── nginx.conf        # (Future) Nginx configuration
│
├── /tests/               # Testing
│   └── test_endpoints.py # API endpoint tests
│
├── /data/               # Local data cache
├── /logs/              # Analysis logs
├── /temp_files/        # Temporary processing files
│
# Configuration Files
├── pyproject.toml      # Python project configuration
├── poetry.lock        # Dependencies lock file
├── .env.template     # Environment template
├── .gitignore       # Git ignore rules
├── .dockerignore    # Docker ignore rules
```

### Key Module Descriptions

#### Core Processing Modules
- **sampler.py**: Implements sophisticated data sampling strategies including time-based filtering, stratified sampling, and reservoir sampling for large datasets.
- **inference_ops.py**: Orchestrates the analysis pipeline with text chunking, parallel summarization, and context preservation.
- **model_ops.py**: Manages model providers and handles fallback mechanisms.
- **embedding_ops.py**: Handles embedding generation and storage with multi-provider support.

#### Data Processing
- **cloud_handler.py**: Manages S3 data operations including batch CSV loading and merging.
- **processing.py**: Provides text preprocessing utilities including normalization and cleaning.
- **stratified_ops.py**: Implements data splitting and stratification logic.

#### API and Frontend
- **api/routes.py**: Defines REST endpoints for query processing and batch operations.
- **chainlit_frontend/app.py**: Provides an interactive UI for experimenting with different providers and configurations.

## Data Flow and Processing Pipeline

### 1. Data Preparation
The pipeline begins with data preparation and sampling:
- Data is loaded from S3 or local sources via `cloud_handler.py`
- `sampler.py` applies filtering and sampling strategies:
  - Time-based filtering for recent/relevant data
  - Stratified sampling for balanced representation
  - Reservoir sampling for large dataset handling
- Text preprocessing via `processing.py` normalizes and cleans the content

### 2. Embedding Generation
Once data is prepared:
- `embedding_ops.py` generates embeddings using the configured provider
- Embeddings are stored for efficient retrieval
- Batch processing optimizes throughput

### 3. Analysis and Inference
The core analysis is orchestrated by `inference_ops.py`:
1. **Chunking**: Text is split into manageable segments
2. **Parallel Processing**: Chunks are analyzed concurrently
3. **Context Preservation**: Relationships between chunks are maintained
4. **Summary Generation**: Final output includes:
   - Core content summary
   - Temporal context (if enabled)
   - Event relationships (if enabled)

### 4. Delivery
Results are available through:
- REST API endpoints for programmatic access
- Interactive Chainlit UI for exploration
- Batch processing for multiple queries

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities

## Development and Testing Workflow

### Local Development Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Unix/MacOS
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.template .env
   # Edit .env with required credentials
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