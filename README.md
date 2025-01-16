# Chanscope Knowledge Agent

## Overview
An advanced query system leveraging multiple AI providers (OpenAI, Grok, Venice) for intelligent text analysis. The system employs a multi-stage pipeline focusing on temporal analysis, event mapping, and forecasting.

## Key Features
- **Multi-Provider Architecture**
  - OpenAI (Primary): GPT-4, text-embedding-3-large
  - Grok (Optional): grok-2-1212, grok-v1-embedding
  - Venice (Optional): llama-3.1-405b, dolphin-2.9.2-qwen2-72b

- **Intelligent Data Processing**
  - Time-based and category-based stratified sampling
  - Efficient large dataset handling with reservoir sampling
  - Automated data chunking and embedding generation

- **Advanced Analysis Pipeline**
  - Context-aware temporal analysis
  - Parallel processing with automatic fallback
  - Event mapping and relationship extraction


The system is designed to process text data through specialized models, each optimized for their specific task. This modular approach allows for flexible model selection while maintaining robust operation through automatic fallback mechanisms.

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

## API Usage

### Basic Query Structure
```python
{
    "query": "Your analysis query",
    "force_refresh": false,
    "batch_size": 100,
    "embedding_provider": "openai",
    "summary_provider": "venice"
}
```

### PowerShell Example
```powershell
# Create request body
$body = @{
    query = "What are the latest developments in AI?"
    force_refresh = $false
    batch_size = 100
    embedding_provider = "openai"
    summary_provider = "openai"
} | ConvertTo-Json

# Make request and get summary
$response = Invoke-RestMethod -Uri "http://localhost:5000/process_query" -Method Post -Body $body -ContentType "application/json"
$response.results.summary

# Optional: View full response
$response | ConvertTo-Json -Depth 10
```

### Bash Example
```bash
# Single query
curl -X POST "http://localhost:5000/process_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "force_refresh": false,
    "batch_size": 100,
    "embedding_provider": "openai",
    "summary_provider": "openai"
  }'

# Batch processing
curl -X POST "http://localhost:5000/batch_process" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What are the latest developments in AI?",
      "How is AI impacting healthcare?"
    ],
    "force_refresh": false,
    "sample_size": 100,
    "embedding_batch_size": 2048,
    "chunk_batch_size": 20,
    "summary_batch_size": 20,
    "embedding_provider": "openai",
    "summary_provider": "openai"
  }'
```

### Output Format
Results include:
- Temporal context and time range
- Thread metrics and patterns
- Content analysis and key claims
- Signal detection and forecasts
- Confidence scores and reliability metrics

## Core Components

### Data Processing Pipeline
- **Sampler**: Advanced filtering and stratification
- **Embeddings**: Multi-provider semantic analysis
- **Inference**: Parallel processing and chunking
- **Analysis**: Context-aware summarization

### Project Structure
```
knowledge_agents/
├── knowledge_agents/     # Core processing modules
├── api/                 # REST API service
├── chainlit_frontend/   # Interactive UI
├── config/             # Configuration files
└── deployment/         # Docker configurations
```

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