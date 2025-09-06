# Chanscope Retrieval

Multi-provider LLM microservice and data pipeline for practical information intelligence over social data (4chan, X). It provides a clean API, a robust ingestion/stratification/embedding workflow, and optional natural-language-to-SQL queries when a database is available.

## Highlights
- Natural language to SQL (NL→SQL) queries when PostgreSQL is available (default in Replit).
- Ingestion → stratification → embedding generation with environment-aware storage.
- Multi‑provider LLM support: OpenAI (required), Grok (optional), Venice (optional).
- FastAPI API with background processing and task management.

## What’s New
- Smarter S3 selection with pagination and filename date-range parsing. Prefers the latest snapshot file that overlaps your retention window and falls back to LastModified when needed. Implemented in `knowledge_agents/data_processing/cloud_handler.py`.
- Interactive NL Query improvements in `scripts/nl_query.py`:
  - Accepts base host, `/api`, `/api/v1`, or full `/api/v1/nl_query` and normalizes
  - Better error messages and dynamic table rendering
  - Respects `API_BASE_URL`; adds `tabulate` dependency
  - Note: The `/api/v1/nl_query` endpoint requires PostgreSQL (e.g., Replit)
- Data wipe utility now supports production and object storage:
  - `scripts/wipe_all_data.py --yes [--database-url ... | --pg-host ... --pg-user ... --pg-password ...]`
  - Optional skips: `--no-kv`, `--no-objects`, `--no-files`
  - Clears PostgreSQL tables, Replit KV, Replit Object Storage artifacts, and file-based artifacts
- Consolidated environment checks into Python: use `python scripts/process_data.py --check`. `scripts/replit_setup.sh` now does lightweight app verification only. Removed `scripts/check_replit_db.py`.

## Architecture (Concise)
- Core orchestrator: `ChanScopeDataManager` controls ingestion, stratified sampling, and embedding generation.
- Storage backends by environment:
  - Replit: PostgreSQL (complete data), Replit Key‑Value (stratified sample), Replit Object Storage (embeddings), Object Storage (process locks).
  - Docker/Local: File‑based CSV/NPZ/JSON with file locks.
- API: FastAPI app (`api.app`) with health, data ops, and NL→SQL endpoints (NL→SQL requires PostgreSQL).
- Scheduling: Optional updates via `scripts/scheduled_update.py` with interval control.

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

## Storage & Environments
- Docker/Local (file‑based):
  - Complete data: `data/complete_data.csv`
  - Stratified sample: `data/stratified/stratified_sample.csv`
  - Embeddings: `data/stratified/embeddings.npz`
  - Locks: file-based
- Replit (database‑backed):
  - Complete data: PostgreSQL tables (`complete_data`, `metadata`)
  - Stratified sample: Replit Key‑Value store
  - Embeddings: Replit Object Storage (.npz)
  - Locks: Object Storage

Environment detection comes from `config/env_loader.detect_environment()` and is used consistently across the codebase.

## Setup

### Docker (Local)
```bash
docker-compose -f deployment/docker-compose.yml build --no-cache
docker-compose -f deployment/docker-compose.yml up -d

# Verify environment detection (should print: docker)
docker exec $(docker ps -q) python -c "from config.env_loader import detect_environment; print(detect_environment())"
```

### Replit
1) Fork to your Replit account and set Secrets:
```
OPENAI_API_KEY=your_key
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_key
S3_BUCKET=your_bucket
```
2) Click Run (Replit starts `uvicorn` and runs `scripts/replit_init.sh`).
3) Optional verification:
```bash
bash scripts/replit_setup.sh            # lightweight app verification
python scripts/process_data.py --check  # consolidated, non‑invasive data/storage check
```
Notes:
- `--check` does not write to PostgreSQL, KV, or Object Storage; safe for empty resources.
- NL→SQL is only available when PostgreSQL is configured (e.g., Replit).

## Data Processing CLI
```bash
# Process all stages (ingestion, stratification, embeddings)
python scripts/process_data.py

# Status only (non‑invasive)
python scripts/process_data.py --check
Wipe dev (Replit development) completely:

# Force refresh all data
python scripts/process_data.py --force-refresh

# Regenerate from existing data
python scripts/process_data.py --regenerate --stratified-only
python scripts/process_data.py --regenerate --embeddings-only

# Bypass process locks (use with caution)
python scripts/process_data.py --ignore-lock
```

## API Quick Start
```bash
# Run the API locally (if not using Replit)
python -m uvicorn api.app:app --host 0.0.0.0 --port 80

# Example NL→SQL (requires PostgreSQL)
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts about Bitcoin from last week",
    "limit": 20
  }'
```

For detailed API routes and request bodies, see `api/README_REQUESTS.md`.

## Refresh Dashboard
- UI: open `http://localhost/refresh` to monitor status, current row count, and control auto-refresh.
- API: the dashboard exposes endpoints under `/refresh/api` (e.g., `/refresh/api/status`).
- CLI control: `python scripts/refresh_control.py status|start|stop|run-once [--interval SEC] [--base http://host/refresh/api]`.
- Metrics shown include total runs, current row count, success rate, average duration, and average rows processed (delta per refresh).

### Auto-Start & Security
- Set `AUTO_REFRESH_MANAGER=true` to auto-start the background refresh loop on startup. Optional intervals: `DATA_REFRESH_INTERVAL` or `REFRESH_INTERVAL` (seconds; default 3600 when using manager auto-start).
- Protect control endpoints with a shared secret by setting `REFRESH_CONTROL_TOKEN`. Then:
  - CLI automatically sends the token from env or via `--token`.
  - Dashboard UI supports `?token=YOUR_TOKEN` in the URL and forwards it to control requests.
  - You can also send `Authorization: Bearer YOUR_TOKEN` or `X-Refresh-Token: YOUR_TOKEN`.

## S3 Ingestion Behavior
- Paginates via `ListObjectsV2` and parses filename date ranges like `*_YYYY-MM-DD_YYYY-MM-DD_*.csv`.
- Prefers the latest snapshot whose end date overlaps the requested window.
- Applies board filters from `SELECT_BOARD`.
- Falls back to LastModified filtering if date ranges are absent.

## Process Locks
- Replit: Object Storage locks ensure single‑instance processing across restarts.
- Docker/Local: File‑based locks with stale lock cleanup.

## Maintenance: Wipe Utilities
Use with extreme caution.
```bash
# Development (Replit): wipe KV, Object Storage, PostgreSQL, and files
python scripts/wipe_all_data.py --yes

# Production DB by full DSN
python scripts/wipe_all_data.py --yes --database-url "postgres://user:pass@host:5432/db"

# Production DB by discrete params
python scripts/wipe_all_data.py --yes --pg-host host --pg-user user --pg-password pass

# Skip scopes
python scripts/wipe_all_data.py --yes --no-kv --no-objects
python scripts/wipe_all_data.py --yes --no-files
```

## Testing
- Tests cover ingestion, embeddings, API endpoints, and the end‑to‑end pipeline.
- See `tests/README_TESTING.md` for running guidance.

## Supported Models
- OpenAI (required): completions and embeddings
- Grok (optional): completions and chunking
- Venice (optional): completions and chunking

## References
- Data Gathering Lambda: https://github.com/joelwk/chanscope-lambda
- Original Chanscope R&D: https://github.com/joelwk/chanscope
- R&D Sandbox: https://github.com/joelwk/knowledge-agents
- Providers: OpenAI, Grok (x.ai), Venice

---
