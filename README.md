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

### Docker Environment Features
- **Multi-Service Architecture**:
  - API service (FastAPI)
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
```

## API and Frontend Integration

### API Endpoints (v1)

All API endpoints are versioned under `/api/v1`. The API provides comprehensive Pydantic model validation, structured error handling, and optimized batch processing.

#### 1. Query Processing
- `POST /api/v1/process_query`: Process a single query with optional batching
```python
# Request
{
    "query": str,                    # Required: Your analysis query
    "force_refresh": bool = False,   # Optional: Force knowledge base refresh
    "skip_embeddings": bool = False, # Optional: Skip embedding generation
    "skip_batching": bool = False,   # Optional: Process immediately vs. batch queue
    "filter_date": str = None,       # Optional: Filter results by date (ISO format)
    "select_board": str = None,      # Optional: Filter by specific board
    "embedding_provider": str = None, # Optional: Override default embedding provider
    "chunk_provider": str = None,    # Optional: Override default chunk provider
    "summary_provider": str = None   # Optional: Override default summary provider
}

# Response (Direct Processing)
{
    "chunks": [
        {
            "content": str,
            "score": float,
            "metadata": Dict[str, Any]  # Optional
        }
    ],
    "summary": str
}

# Response (Batch Processing)
{
    "status": "queued",
    "batch_id": str,
    "message": str,
    "position": int,
    "eta_seconds": int
}
```

- `POST /api/v1/batch_process`: Process multiple queries in batch
```python
# Request
{
    "queries": List[str],           # Required: List of queries to process
    "force_refresh": bool = False,  # Optional: Force refresh
    "config": {                     # Optional: Configuration overrides
        "embedding_batch_size": int = 25,
        "chunk_batch_size": int = 25000,
        "summary_batch_size": int = 50,
        "embedding_provider": str = "openai",
        "chunk_provider": str = "openai",
        "summary_provider": str = "openai",
        "select_board": str = None
    }
}

# Response
{
    "task_id": str,
    "status": "processing",
    "message": str,
    "total_queries": int,
    "estimated_completion_time": str  # ISO format datetime
}
```

- `GET /api/v1/batch_status/{task_id}`: Check batch processing status
```python
# Response
{
    "status": str,  # "initializing", "processing", "completed", "failed"
    "message": str,
    "progress": {
        "total": int,
        "completed": int,
        "percent": float
    },
    "results": List[Dict] if completed,
    "errors": List[Dict] if any errors
}
```

- `GET /api/v1/process_recent_query`: Process data from the last 6 hours
```python
# Request Parameters
select_board: str = None  # Optional query parameter to filter by board

# Response
{
    "status": str,  # "queued" or direct results
    "batch_id": str if queued,
    "time_range": {
        "start": str,  # ISO format datetime
        "end": str     # ISO format datetime
    },
    # If direct processing:
    "chunks": List[Dict[str, Any]],
    "summary": str
}
```

#### 2. Embedding Generation
- `POST /api/v1/trigger_embedding_generation`: Trigger background embedding generation
```python
# Response
{
    "status": "started",
    "message": str,
    "task_id": str
}
```

- `GET /api/v1/embedding_status`: Check embedding generation status
```python
# Response
{
    "status": str,  # "idle", "processing", "completed", "failed"
    "progress": {
        "current": int,
        "total": int,
        "percent": float
    },
    "start_time": str,  # ISO format datetime
    "estimated_completion_time": str if processing
}
```

#### 3. Data Management
- `POST /api/v1/stratify`: Stratify data for processing
```python
# Response
{
    "status": "success",
    "message": str,
    "data": {
        "stratified_rows": int,
        "stratified_file": str
    }
}
```

#### 4. Health Check Endpoints
- `GET /api/v1/health`: Basic health check with version info
```python
# Response
{
    "status": str,
    "message": str,
    "timestamp": datetime,
    "environment": {
        "ENV": str,
        "DEBUG": bool
    }
}
```

- `GET /api/v1/health/connections`: Check all service connections
```python
# Response
{
    "services": {
        "openai": Dict[str, Any],
        # Other providers as configured
    }
}
```

- `GET /api/v1/health/s3`: Check S3 connection status
```python
# Response
{
    "s3_status": str,
    "bucket_access": bool,
    "bucket_name": str,
    "bucket_details": Dict[str, Any],
    "aws_region": str,
    "latency_ms": float
}
```

- `GET /api/v1/health/provider/{provider}`: Check specific provider status
```python
# Response
{
    "status": str,
    "provider": str,
    "latency_ms": float
}
```

- `GET /api/v1/health/all`: Check all configured providers
```python
# Response
{
    "status": "completed",
    "providers": {
        "openai": Dict[str, Any],
        # Other providers as configured
    },
    "latency_ms": float
}
```

- `GET /api/v1/health/cache`: Check cache status
```python
# Response
{
    "status": str,
    "stats": {
        "hits": int,
        "misses": int,
        "errors": int,
        "hit_ratio": float
    },
    "backend": str,
    "ttl": int
}
```

- `GET /api/v1/health/embeddings`: Check embedding status
```python
# Response
{
    "status": str,
    "embedding_stats": {
        "total_embeddings": int,
        "missing_embeddings": int,
        "embedding_coverage": float
    },
    "providers": List[str]
}
```

#### 5. Debug Endpoints
- `GET /api/v1/debug/routes`: List all available routes
- `GET /api/v1/debug/request`: Echo request details for debugging

### Error Handling

All endpoints use a consistent error handling structure:

```python
{
    "error": str,           # Error code
    "message": str,         # Human-readable message
    "status_code": int,     # HTTP status code
    "details": Dict,        # Additional error details
    "timestamp": str        # ISO format datetime
}
```

Error types include:
- `VALIDATION_ERROR`: Request validation failures
- `CONFIGURATION_ERROR`: System configuration issues
- `PROCESSING_ERROR`: Data processing failures
- `PROVIDER_ERROR`: AI provider communication issues

### Example Usage

#### 1. Using PowerShell
```powershell
# Process a single query
$body = @{
    query = "What are the latest developments in AI?"
    force_refresh = $false
    skip_embeddings = $false
    skip_batching = $true
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:5000/api/v1/process_query" -Method Post -Body $body -ContentType "application/json"

# View results
$response.summary
```

#### 2. Using Bash/cURL
```bash
# Single query with direct processing
curl -X POST "http://localhost:5000/api/v1/process_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "force_refresh": false,
    "skip_embeddings": false,
    "skip_batching": true
  }'

# Batch processing
curl -X POST "http://localhost:5000/api/v1/batch_process" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What are the latest developments in AI?", "What are people saying about cryptocurrency?"],
    "config": {
      "embedding_provider": "openai",
      "select_board": "biz"
    }
  }'

# Health check
curl "http://localhost:5000/api/v1/health"

# Recent data query
curl "http://localhost:5000/api/v1/process_recent_query?select_board=biz"
```

#### 3. Using Python Requests
```python
import requests
import time

# Configuration
API_BASE = "http://localhost:5000/api/v1"

# Process a query with batching
response = requests.post(
    f"{API_BASE}/process_query",
    json={
        "query": "What are the latest developments in AI?",
        "force_refresh": False,
        "skip_embeddings": False,
        "skip_batching": False
    }
)

# Check response
if response.status_code == 200:
    data = response.json()
    
    # If queued for batch processing
    if "batch_id" in data:
        batch_id = data["batch_id"]
        print(f"Query queued for processing. Batch ID: {batch_id}")
        print(f"Position in queue: {data['position']}")
        print(f"Estimated wait time: {data['eta_seconds']} seconds")
        
        # Poll for results
        while True:
            time.sleep(5)  # Wait 5 seconds between checks
            status_response = requests.get(f"{API_BASE}/batch_status/{batch_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                print("Processing complete!")
                print(status_data["results"][0]["summary"])
                break
            elif status_data["status"] == "failed":
                print("Processing failed:", status_data["message"])
                break
            else:
                print(f"Status: {status_data['status']} - {status_data['progress']['percent']}% complete")
    
    # If processed immediately
    elif "summary" in data:
        print("Query processed immediately:")
        print(data["summary"])
```

#### 4. Integration with AI Agents
```python
# Example of how an AI agent might use the Knowledge Agent API
class Agent4ChanAnalyzer:
    def __init__(self, api_base="http://localhost:5000/api/v1"):
        self.api_base = api_base
        self.memory = []  # Simple memory store
        
    async def analyze_topic(self, topic, board=None):
        """Agent analyzes a specific topic on 4chan"""
        query = f"What are people saying about {topic} in the last 6 hours?"
        
        # Call Knowledge Agent API
        response = requests.post(
            f"{self.api_base}/process_query",
            json={
                "query": query,
                "select_board": board,
                "skip_batching": True  # Get immediate results
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store in agent memory
            self.memory.append({
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "summary": data["summary"],
                "source_chunks": len(data["chunks"])
            })
            
            # Agent processes the information
            return self._process_information(data["summary"], topic)
        
        return "Failed to retrieve information"
    
    def _process_information(self, summary, topic):
        """Agent's internal processing of information"""
        # This would contain the agent's own logic for processing the data
        return f"Analysis of {topic}: {summary}"
```

## Project Structure
```
knowledge_agents/
├── api/                # FastAPI application
│   ├── routes.py      # API endpoints
│   ├── models.py      # Response models
│   ├── errors.py      # Error handling
│   ├── cache.py       # Caching system
│   └── app.py         # Application setup
├── config/            # Configuration files
├── data/              # Data storage
├── deployment/        # Deployment configurations
├── knowledge_agents/  # Core business logic
│   ├── model_ops.py   # Model operations
│   ├── data_ops.py    # Data operations
│   ├── embedding_ops.py # Embedding generation
│   ├── inference_ops.py # Inference operations
│   └── data_processing/ # Data processing utilities
├── logs/             # Application logs
├── scripts/          # Utility scripts
├── tests/            # Test suite
└── temp_files/       # Temporary file storage
```

### Key Components

#### 1. Core Processing (`knowledge_agents/`)
- Data processing pipeline and utilities
- Model operations and inference
- Embedding and stratification logic
- Utility functions and configurations

#### 2. API Service (`api/`)
- REST API implementation with FastAPI
- Structured error handling system
- Efficient batch processing
- Comprehensive health checks
- Caching system for query results
- Background task management

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

# Start scheduled updates (optional)
python scripts/scheduled_update.py
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

#### 3. Scheduled Updates
The system includes an automated data update mechanism:
- Hourly data refresh from S3
- Incremental updates to minimize processing
- Automatic state management
- Configurable time windows
- Error handling and retry logic

To run the scheduled update service:
```bash
# Using the provided script (runs continuously)
./scripts/replit/scheduled_update.sh

# Or manually trigger an update
python scripts/scheduled_update.py
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
curl http://localhost:5000/api/v1/health

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

### Data Processing Settings (Optional, with defaults)
- `TIME_COLUMN`: Column name for timestamps (Default: 'posted_date_time')
- `STRATA_COLUMN`: Column name for stratification (Optional)
- `FREQ`: Frequency for time-based operations (Default: 'H')
- `FILTER_DATE`: Date to filter data from (Optional)
- `SELECT_BOARD`: Filter data by specific board (Optional)
- `PADDING_ENABLED`: Enable text padding (Default: false)
- `CONTRACTION_MAPPING_ENABLED`: Enable contraction mapping (Default: false)
- `NON_ALPHA_NUMERIC_ENABLED`: Enable non-alphanumeric processing (Default: false)

### AWS Settings 
- `AWS_ACCESS_KEY_ID`: AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key
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
  - Stratification Chunk Size: 2500
  - Processing Chunk Size: 25000
  - Sample Size: 1500
  - Max Tokens: 4096
  - Cache Enabled: true
  - Max Workers: 4
- **Update Schedule**:
  - Hourly data refresh
  - 6-hour rolling window for real-time queries
  - 30-day data retention window