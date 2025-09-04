# Overview

Chanscope Retrieval is a multi-provider LLM microservice and data pipeline designed for practical information intelligence over social data (4chan, X). The system provides natural language to SQL queries when PostgreSQL is available, a robust ingestion/stratification/embedding workflow, and multi-provider LLM support. The application follows a biologically-inspired architecture with distinct yet interconnected processing stages, supporting both file-based storage (Docker/local) and database storage (Replit environments).

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Orchestration
The system centers around the `ChanScopeDataManager` which controls the complete data lifecycle: ingestion from S3 sources, stratified sampling for representative subsets, and embedding generation. This manager implements the Chanscope approach with specific behaviors based on the `force_refresh` flag and environment-aware storage selection.

## Storage Architecture
**Environment-Based Storage Strategy:**
- **Replit Environment**: PostgreSQL for complete datasets, Replit Key-Value store for stratified samples, Replit Object Storage for embeddings and process locks
- **Docker/Local Environment**: File-based storage using CSV for data, NPZ for embeddings, JSON for metadata, with file-based locking mechanisms

**Storage Abstraction**: Abstract interfaces (`CompleteDataStorage`, `StratifiedSampleStorage`, `EmbeddingStorage`, `StateManager`) provide unified access patterns across different storage backends.

## API Layer
**FastAPI Application** with comprehensive endpoint coverage:
- Health monitoring and cache metrics endpoints
- Data processing orchestration endpoints (`/process_query`, `/batch_process`)
- Natural language to SQL conversion (`/api/v1/nl_query`) requiring PostgreSQL
- Background task management with proper lifecycle handling

## LLM Integration Architecture
**Multi-Provider Support**: OpenAI (required), Grok (optional), Venice (optional) with configurable fallback chains and retry mechanisms.

**Three-Stage SQL Generation Pipeline**:
1. **Instruction Enhancement**: First LLM refines natural language into structured instructions
2. **SQL Generation**: Second LLM converts instructions to parameterized SQL
3. **SQL Validation**: Same LLM validates for security and correctness

**Hybrid Query Processing**: Template matching for common patterns with LLM fallback for complex queries, combined with result caching for performance.

## Data Processing Pipeline
**Stratified Sampling**: Creates representative subsets preserving temporal and categorical distributions, with configurable sample sizes and intelligent refresh policies.

**Embedding Generation**: Supports both deterministic text chunking and LLM-based chunking with robust fallback mechanisms. Implements batch processing with configurable sizes and FAISS-optimized similarity search.

**Text Validation**: Comprehensive validation throughout the pipeline ensuring data quality and consistency.

## Concurrency and Scheduling
**Process Lock Management**: Cross-environment process locking using either Replit Object Storage or file-based locks to prevent concurrent data processing conflicts.

**Scheduled Updates**: Optional background scheduling with configurable intervals, comprehensive error handling, and automatic recovery mechanisms.

# External Dependencies

## Cloud Storage Services
- **AWS S3**: Primary data source for social media content ingestion
- **Replit Object Storage**: Embedding storage and process coordination in Replit environments
- **Google Cloud Storage**: Secondary storage option with credential patching for compatibility

## Database Systems
- **PostgreSQL**: Complete dataset storage in Replit, enables natural language SQL features
- **Replit Key-Value Store**: Stratified sample storage in Replit environments

## AI/ML Services
- **OpenAI API**: Primary LLM provider for embeddings, SQL generation, and text processing
- **Grok API**: Optional secondary LLM provider with fallback capabilities
- **Venice AI**: Optional specialized LLM provider for SQL generation pipeline

## Search and Analytics
- **FAISS**: High-performance similarity search with CPU optimization
- **scikit-learn**: Machine learning utilities for data stratification and analysis
- **Transformers/Hugging Face**: Model loading and tokenization support

## Development and Deployment
- **FastAPI/Uvicorn**: Web framework and ASGI server
- **Docker**: Containerization for local and production deployments
- **Replit**: Cloud development and hosting platform integration