# Chanscope Knowledge Agent

## Description
A knowledge processing system that leverages multiple AI providers (OpenAI, Grok, Venice) to analyze and generate insights from text data. The system employs a three-stage pipeline:

1. **Embedding Generation**: Creates semantic embeddings for efficient text search and retrieval
2. **Chunk Analysis**: Processes text segments to extract key information and context
3. **Summary Generation**: Produces comprehensive summaries with temporal analysis and forecasting

Key features:
- Multiple model provider support with automatic fallback mechanisms
- Temporal-aware analysis for tracking information evolution over time
- Batch processing capabilities for multiple queries
- REST API endpoints for easy integration
- Docker containerization for simplified deployment
- Configurable model selection for each pipeline stage

The system is designed to process text data through a series of specialized models, each optimized for their specific task (embeddings, chunk analysis, or summarization). This modular approach allows for flexible model selection while maintaining robust operation through automatic fallback mechanisms.

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
1. Build and start the service:
```bash
docker-compose up --build -d
```

2. Verify the service is running:
```bash
curl http://localhost:5000/health
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
├── data/                  # Data storage directory
│   ├── all_data.csv      # Combined raw data
│   ├── stratified/       # Stratified data samples
│   └── knowledge_base.csv # Processed embeddings
├── logs/                  # Application logs
├── knowledge_agents/      # Main package directory
│   ├── model_ops.py      # Model operations
│   ├── data_ops.py       # Data processing
│   ├── inference_ops.py  # Inference pipeline
│   └── prompt.yaml       # System prompts
├── .env.template         # Environment template
├── docker-compose.yml    # Docker configuration
└── app.py               # Flask application
```

## References
- Data Gathering Lambda: [chanscope-lambda](https://github.com/joelwk/chanscope-lambda)
- Original Chanscope R&D: [Chanscope](https://github.com/joelwk/chanscope)
- R&D Sandbox Repository: [knowledge-agents](https://github.com/joelwk/knowledge-agents)
- Inspiration for Prompt Engineering Approach: [Temporal-Aware Language Models for Temporal Knowledge Graph Question Answering](https://arxiv.org/pdf/2410.18959) - Used for designing temporal-aware prompts and multimodal forecasting capabilities