# Chanscope Retrieval

Multi-provider LLM microservice and data pipeline for practical information intelligence over social data (4chan, X). Provides a clean API, robust ingestion/stratification/embedding workflow, and optional natural-language-to-SQL queries when a database is available.

## Features

- **Natural language to SQL (NL→SQL)** queries when PostgreSQL is available (default in Replit)
- **Data pipeline** with ingestion → stratification → embedding generation
- **Multi-provider LLM support**: OpenAI (required), Grok (optional), Venice (optional)
- **FastAPI** with background processing and task management
- **Environment-aware storage** backends for Docker, Local, and Replit deployments

## System Architecture

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
│  ┌────────────┐ │         │  │     Chanscope      │  │     │
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

## Storage Backends

### Docker/Local (File-based)
- Complete data: `data/complete_data.csv`
- Stratified sample: `data/stratified/stratified_sample.csv`
- Embeddings: `data/stratified/embeddings.npz`
- Locks: file-based

### Replit (Database-backed)
- Complete data: PostgreSQL tables
- Stratified sample: Replit Key-Value store
- Embeddings: Replit Object Storage
- Locks: Object Storage

## Quick Start

### Docker Setup
```bash
docker-compose -f deployment/docker-compose.yml build --no-cache
docker-compose -f deployment/docker-compose.yml up -d
```

### Replit Setup
1. Fork to your Replit account
2. Set Secrets:
   ```
   OPENAI_API_KEY=your_key
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_key
   S3_BUCKET=your_bucket
   ```
3. Click Run

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python -m uvicorn api.app:app --host 0.0.0.0 --port 80
```

## Data Processing

```bash
# Process all stages
python scripts/process_data.py

# Check status
python scripts/process_data.py --check

# Force refresh
python scripts/process_data.py --force-refresh

# Regenerate components
python scripts/process_data.py --regenerate --stratified-only
python scripts/process_data.py --regenerate --embeddings-only
```

## API Usage

```bash
# Natural Language Query (requires PostgreSQL)
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts about Bitcoin from last week",
    "limit": 20
  }'

# Standard Query
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Investment opportunities in renewable energy"
  }'

# Background Processing
curl -X POST "http://localhost/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bitcoin Strategic Reserve",
    "use_background": true,
    "task_id": "bitcoin_analysis_123"
  }'
```

## Key Components

- **ChanScopeDataManager**: Central orchestrator for data operations
- **StorageFactory**: Environment-aware storage backend selection
- **KnowledgeAgent**: Singleton LLM service for embeddings and completions
- **LLMSQLGenerator**: Natural language to SQL conversion
- **FastAPI Application**: RESTful API with background task support

## Environment Variables

See `.env.template` for complete list. Key variables:
- `OPENAI_API_KEY`: Required for embeddings and completions
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: S3 access
- `DATA_RETENTION_DAYS`: Data retention period (default: 30)
- `GROK_API_KEY`, `VENICE_API_KEY`: Optional providers

## Data Wipe Utility

```bash
# Development
python scripts/wipe_all_data.py --yes

# Production with database URL
python scripts/wipe_all_data.py --yes --database-url "postgres://..."

# Selective wipe
python scripts/wipe_all_data.py --yes --no-kv --no-objects
```

## Documentation

- [Deployment Guide](deployment/README_DEPLOYMENT.md)
- [API Reference](api/README_REQUESTS.md)
- [Testing Guide](tests/README_TESTING.md)
- [Implementation Details](docs/chanscope_implementation.md)

## References

- [Data Collection Lambda](https://github.com/joelwk/chanscope-lambda)
- [Original Chanscope R&D](https://github.com/joelwk/chanscope)
- [Knowledge Agents Sandbox](https://github.com/joelwk/knowledge-agents)