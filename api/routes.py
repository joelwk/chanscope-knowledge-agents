"""API routes and endpoint handlers."""
import time
import traceback
import os
import yaml
import asyncio
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

import json
import shutil
from filelock import FileLock
import pytz
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import secrets
import re
import copy
import numpy as np

from knowledge_agents.model_ops import (
    ModelProvider, 
    KnowledgeAgent, 
    ModelProviderError,
    ModelConfigurationError,
    ModelOperationError,
    ModelOperation,
    ModelConfig,
    load_config)

from knowledge_agents.embedding_ops import get_agent
from knowledge_agents.data_processing.cloud_handler import S3Handler
from knowledge_agents.data_ops import DataConfig, DataOperations
from knowledge_agents.inference_ops import process_multiple_queries_efficient, process_query
from api.errors import ProcessingError, APIError, ConfigurationError, ValidationError

from config.base_settings import get_base_settings
from config.settings import Config
from config.config_utils import (
    QueryRequest,
    BatchQueryRequest,
    build_unified_config)

from config.logging_config import get_logger
from api.models import HealthResponse, StratificationResponse, log_endpoint_call
from api.cache import CACHE_HITS, CACHE_MISSES, CACHE_ERRORS, cache

from config.env_loader import is_replit_environment, get_replit_paths, detect_environment
from knowledge_agents.utils import save_query_output

# Import the LLM-based SQL generator
from knowledge_agents.llm_sql_generator import LLMSQLGenerator, NLQueryParsingError

logger = get_logger(__name__)
router = APIRouter()

# Global variables for tracking tasks and results
_background_tasks: Dict[str, Dict[str, Any]] = {}
_batch_results: Dict[str, Any] = {}
_query_batch: List[Dict[str, Any]] = []
_BATCH_SIZE = Config.get_model_settings()['embedding_batch_size']  # Batch size from configuration
_BATCH_WAIT_TIME = 2  # Seconds to wait for batch accumulation
_last_batch_processing_time = 0  # Track processing time for ETA estimates
_tasks_lock = asyncio.Lock()
_embedding_task_key = "embedding_generation"

task_response_queue: asyncio.Queue = asyncio.Queue()

# Set up periodic cleanup task for memory and disk
async def _start_periodic_cleanup():
    """Run cleanup at regular intervals."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run cleanup every hour
            await _cleanup_old_results()
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            # Continue running even if an iteration fails
            await asyncio.sleep(3600)  # Wait before retrying

# Function to start cleanup task - call this during app startup
async def start_cleanup_task():
    """Start the periodic cleanup task when the API starts."""
    logger.info("Starting periodic cleanup task for query results")
    asyncio.create_task(_start_periodic_cleanup())

# Import run_inference lazily to avoid circular import
def get_run_inference():
    from knowledge_agents.run import run_inference
    return run_inference

# Dependency for DataOperations
async def get_data_ops() -> DataOperations:
    data_config = DataConfig.from_config()
    return DataOperations(data_config)

# Dependency for KnowledgeAgent
async def get_knowledge_agent() -> KnowledgeAgent:
    """Get the KnowledgeAgent singleton instance."""
    return await get_agent()

# Shared API documentation structure
API_DOCS = {
    "name": "Knowledge Agents API",
    "version": "1.0",
    "documentation": {
        "root": "/",
        "health": "/health",
        "healthz": "/healthz",
        "health_replit": "/health_replit",
        "health_connections": "/health/connections",
        "health_s3": "/health/s3",
        "health_provider": "/health/provider/{provider}",
        "health_all_providers": "/health/all",
        "health_cache": "/health/cache",
        "health_embeddings": "/health/embeddings",
        "process_query": "/query",
        "batch_process": "/batch_process",
        "batch_status": "/batch_status/{task_id}",
        "stratify": "/stratify",
        "process_recent_query": "/process_recent_query",
        "trigger_embedding_generation": "/trigger_embedding_generation",
        "embedding_status": "/embedding_status",
        "debug_routes": "/debug/routes",
        "debug_request": "/debug/request",
        "nl_query": "/nl_query"
    }
}

def get_api_docs(with_prefix: str = ""):
    docs = API_DOCS.copy()
    if with_prefix:
        docs["documentation"] = {key: f"{with_prefix}{path}" for key, path in docs["documentation"].items()}
    return docs

@router.get("/", response_model=dict)
async def root():
    """Return API documentation."""
    return get_api_docs("/api")

@router.get("/health")
async def health_check():
    """Super lightweight health check endpoint for Replit deployment.
    This must be extremely fast and shouldn't depend on any database or storage operations."""
    try:
        # Use a very minimal response with just the essential information
        return {
            "status": "ok",
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "environment": os.getenv("REPLIT_ENV", "development")
        }
    except Exception as e:
        # Even on error, return a 200 OK to pass health checks
        logger.error(f"Error in health check: {e}")
        return {"status": "ok", "error_handled": True}

@router.get("/healthz", include_in_schema=False)
async def healthz():
    """Simple health check endpoint for Replit's health check system."""
    return {
        "status": "ok", 
        "ready": True
    }

@router.get("/health_replit", response_model=dict)
async def health_check_replit():
    """Extended health check with Replit-specific info."""
    from config.env_loader import is_replit_environment
    from config.settings import Config

    # Get Replit environment variables from centralized config
    api_settings = Config.get_api_settings()

    # Get Replit environment variables
    replit_env = os.getenv('REPLIT_ENV', '')
    replit_id = os.getenv('REPL_ID', '')
    replit_slug = os.getenv('REPL_SLUG', '')
    replit_owner = os.getenv('REPL_OWNER', '')
    replit_dev_domain = os.getenv('REPLIT_DEV_DOMAIN', '')

    # Construct service URL based on available info
    if replit_dev_domain:
        service_url = f"https://{replit_dev_domain}"
    elif replit_id:
        service_url = f"https://{replit_id}.id.repl.co"
    elif replit_slug and replit_owner:
        service_url = f"https://{replit_slug}.{replit_owner}.repl.co"
    else:
        service_url = None

    # Get port configuration from Config
    port = api_settings.get('port', 80)
    api_port = port
    host = api_settings.get('host', '0.0.0.0')

    # Check if running in Replit environment
    is_replit_env = is_replit_environment()

    # Get additional environment information
    environment_vars = {
        "is_replit": is_replit_env,
        "replit_env": replit_env,
        "repl_id": replit_id,
        "repl_slug": replit_slug,
        "repl_owner": replit_owner,
        "replit_dev_domain": replit_dev_domain,
        "python_path": os.getenv('PYTHONPATH', ''),
        "fastapi_env": api_settings.get('fastapi_env', ''),
        "fastapi_debug": api_settings.get('fastapi_debug', '')
    }

    # Get service configuration
    service_config = {
        "url": service_url,
        "port": port,
        "api_port": api_port,
        "host": host,
        "api_url": f"{service_url}/api" if service_url else None,
        "api_v1_url": f"{service_url}/api/v1" if service_url else None,
        "api_base_path": api_settings.get('base_path', '/api/v1'),
        "docs_url": f"{service_url}/docs" if service_url else None
    }

    # Get system information
    import platform
    import sys

    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "memory": f"{os.getenv('REPL_MEMORY', 'unknown')} MB"
    }

    # Check data directories and permissions
    paths = get_replit_paths()
    data_status = {
        "root_data_exists": os.path.exists(paths["root_data_path"]),
        "stratified_data_exists": os.path.exists(paths["stratified_path"]),
        "logs_exists": os.path.exists(paths["logs_path"]),
        "temp_exists": os.path.exists(paths["temp_path"])
    }

    return {
        "status": "healthy",
        "message": "Replit Service is running",
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "environment": environment_vars,
        "service": service_config,
        "system": system_info,
        "data": data_status,
        "version": "1.0.0"
    }

@router.get("/health/connections", response_model=dict)
async def service_connections(agent: KnowledgeAgent = Depends(get_agent)):
    """Check connections for external services."""
    # Example check for OpenAI
    async def check_openai():
        client = await agent._get_client(ModelProvider.OPENAI)
        return await client.models.list()
    try:
        openai_status = await check_openai()
    except Exception as e:
        openai_status = str(e)
    return {"services": {"openai": openai_status}}

@router.get("/health/s3", response_model=dict)
async def s3_health():
    """Check S3 connection and bucket access."""
    start_time = time.time()
    try:
        s3_handler = S3Handler()
        import aioboto3
        session = aioboto3.Session()
        async with session.client(
            's3',
            region_name=s3_handler.region_name,
            aws_access_key_id=s3_handler.aws_access_key_id,
            aws_secret_access_key=s3_handler.aws_secret_access_key
        ) as s3_client:
            try:
                await s3_client.head_bucket(Bucket=s3_handler.bucket_name)
                bucket_exists = True
            except s3_client.exceptions.ClientError:
                bucket_exists = False

            bucket_details = {}
            if bucket_exists:
                try:
                    response = await s3_client.list_objects_v2(
                        Bucket=s3_handler.bucket_name,
                        Prefix=s3_handler.bucket_prefix,
                        MaxKeys=1
                    )
                    bucket_details = {
                        "prefix": s3_handler.bucket_prefix,
                        "region": s3_handler.region_name,
                        "has_contents": 'Contents' in response
                    }
                except Exception as e:
                    logger.warning(f"Could not fetch bucket details: {str(e)}")
        return {
            "s3_status": "connected" if bucket_exists else "bucket_not_found",
            "bucket_access": bucket_exists,
            "bucket_name": s3_handler.bucket_name,
            "bucket_details": bucket_details,
            "aws_region": s3_handler.region_name,
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"S3 health check error: {str(e)}")
        error_details = {
            "message": str(e),
            "type": e.__class__.__name__
        }
        return JSONResponse(
            status_code=500,
            content={
                "s3_status": "error",
                "bucket_access": False,
                "error": error_details,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        )

@router.get("/health/provider/{provider}", response_model=dict)
async def provider_health(
    provider: str,
    agent: KnowledgeAgent = Depends(get_agent)
) -> Dict[str, Any]:
    """Check health for a specific provider with proper error handling."""
    start_time = time.time()
    try:
        provider_enum = ModelProvider.from_str(provider)
        client = await agent._get_client(provider_enum)

        # Provider-specific health checks
        response = {}
        if provider_enum == ModelProvider.OPENAI:
            models = await client.models.list()
            response = {
                "status": "healthy",
                "provider": provider,
                "models_available": len(models.data),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        elif provider_enum == ModelProvider.GROK:
            response = {
                "status": "healthy",
                "provider": provider,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        elif provider_enum == ModelProvider.VENICE:
            response = {
                "status": "healthy",
                "provider": provider,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        elif provider_enum == ModelProvider.OPENROUTER:
            response = {
                "status": "healthy",
                "provider": provider,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        else:
            raise ValidationError(
                message=f"Unsupported provider: {provider}",
                field="provider",
                value=provider
            )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_endpoint_call(
            logger=logger,
            endpoint=f"/health/provider/{provider}",
            method="GET",
            duration_ms=duration_ms,
            params={"provider": provider}
        )

        return response

    except ModelProviderError as e:
        error = APIError(
            message=f"Provider error for {provider}",
            status_code=400,
            error_code="PROVIDER_ERROR",
            details={
                "provider": provider,
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        )
        error.log_error(logger)
        return JSONResponse(
            status_code=error.status_code,
            content=error.to_dict()
        )
    except ValidationError as e:
        e.log_error(logger)
        return JSONResponse(
            status_code=e.status_code,
            content=e.to_dict()
        )
    except Exception as e:
        error = APIError(
            message=f"Health check failed for provider {provider}",
            status_code=500,
            error_code="PROVIDER_HEALTH_CHECK_ERROR",
            details={
                "provider": provider,
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        )
        error.log_error(logger)
        return JSONResponse(
            status_code=error.status_code,
            content=error.to_dict()
        )

@router.get("/health/all", response_model=dict)
async def all_providers_health(
    agent: KnowledgeAgent = Depends(get_agent)
) -> Dict[str, Any]:
    """Check health for all configured providers."""
    start_time = time.time()
    results = {}

    for provider in ModelProvider:
        try:
            client = await agent._get_client(provider)
            if provider == ModelProvider.OPENAI:
                models = await client.models.list()
                results[provider.value] = {
                    "status": "healthy",
                    "models_available": len(models.data)
                }
            else:
                results[provider.value] = {
                    "status": "healthy"
                }
        except Exception as e:
            logger.error(f"Health check failed for {provider}: {str(e)}")
            results[provider.value] = {
                "status": "error",
                "error": str(e)
            }

    return {
        "status": "completed",
        "providers": results,
        "latency_ms": round((time.time() - start_time) * 1000, 2)
    }

@router.post("/stratify", response_model=StratificationResponse)
async def stratify_data(
    data_ops: DataOperations = Depends(get_data_ops)
) -> StratificationResponse:
    try:
        paths = Config.get_paths()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()

        status = data_ops.verify_data_structure()
        if not status.get('complete_data', False):
            data_ops.update_complete_data()
            status = data_ops.verify_data_structure()
            if not status.get('complete_data', False):
                raise HTTPException(status_code=500, detail="Failed to create complete dataset.")

        complete_data_path = Path(paths['root_data_path']) / 'complete_data.csv'
        complete_data = pd.read_csv(complete_data_path)
        if complete_data.empty:
            raise HTTPException(status_code=400, detail="Complete dataset is empty.")

        from knowledge_agents.data_processing.sampler import Sampler
        sampler = Sampler(
            time_column=column_settings['time_column'],
            strata_column=column_settings['strata_column'],
            initial_sample_size=sample_settings['default_sample_size']
        )
        stratified_data = sampler.stratified_sample(complete_data)
        stratified_file = Path(paths['stratified']) / "stratified_sample.csv"
        stratified_file.parent.mkdir(parents=True, exist_ok=True)
        stratified_data.to_csv(stratified_file, index=False)
        return StratificationResponse(
            status="success",
            message='Stratification completed successfully',
            data={
                "stratified_rows": len(stratified_data),
                "stratified_file": str(stratified_file)})
    except Exception as e:
        logger.error(f"Error during stratification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def load_stratified_data(
    stratified_path: str,
) -> pd.DataFrame:
    """Load and merge stratified data with embeddings.

    Args:
        stratified_path: Path to the stratified data directory

    Returns:
        DataFrame containing merged stratified data with embeddings
    """
    from knowledge_agents.embedding_ops import merge_articles_and_embeddings
    from pathlib import Path

    # Log input path
    logger.info(f"Loading stratified data from base path: {stratified_path}")

    # Construct full paths - all files should be in the same stratified_path directory
    stratified_file = Path(stratified_path) / 'stratified_sample.csv'
    embeddings_file = Path(stratified_path) / 'embeddings.npz'
    thread_id_file = Path(stratified_path) / 'thread_id_map.json'

    # Verify files exist
    logger.info(f"Checking stratified data files:")
    logger.info(f"  Stratified file: {stratified_file} (exists: {stratified_file.exists()})")
    logger.info(f"  Embeddings file: {embeddings_file} (exists: {embeddings_file.exists()})")
    logger.info(f"  Thread ID file: {thread_id_file} (exists: {thread_id_file.exists()})")

    # Load and merge the data
    library_df = await merge_articles_and_embeddings(
        stratified_file, embeddings_file, thread_id_file
    )

    logger.info(f"Loaded stratified data: {library_df.shape[0]} rows, {library_df.shape[1]} columns")

    return library_df

@router.post("/query", response_model=Dict[str, Any])
async def base_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Process a single query and return results."""
    start_time = time.time()
    task_id = None

    try:
        # Validate request
        if not request.query or not isinstance(request.query, str):
            raise ValidationError(
                message="Invalid query format",
                field="query",
                value=request.query
            )

        # Process user-provided task_id or generate one
        if request.task_id:
            # Validate user-provided task ID
            if not _is_valid_task_id(request.task_id):
                raise ValidationError(
                    message="Invalid task ID format. Task IDs must be 4-64 characters and contain only alphanumeric characters, underscores, and hyphens.",
                    field="task_id",
                    value=request.task_id
                )

            # Check for collisions
            if _check_task_id_collision(request.task_id):
                raise ValidationError(
                    message="Task ID already exists. Please provide a unique task ID.",
                    field="task_id",
                    value=request.task_id
                )

            task_id = request.task_id
            logger.info(f"Using user-provided task ID: {task_id}")
        else:
            # Generate task ID
            task_id = _generate_task_id(prefix="query")
            logger.info(f"Generated task ID: {task_id}")

        logger.info(f"Processing query request {task_id}: {request.query[:50]}...")

        # Get required services
        agent = await get_agent()
        data_ops = await get_data_ops()
        config = build_unified_config(request)

        if request.use_background:
            # Add task to background processing
            background_tasks.add_task(
                _process_single_query,
                task_id=task_id,
                query=request.query,
                agent=agent,
                config=config,
                use_batching=not request.skip_batching,
                data_ops=data_ops,
                force_refresh=request.force_refresh,
                skip_embeddings=request.skip_embeddings,
                character_slug=request.character_slug
            )

            return {
                "status": "processing",
                "task_id": task_id,
                "message": "Query processing started in background"
            }
        else:
            # Process query synchronously
            try:
                # Initialize task status
                async with _tasks_lock:
                    _background_tasks[task_id] = {
                        "status": "processing",
                        "timestamp": time.time(),
                        "query": request.query,
                        "progress": 0
                    }

                # Process query
                # Check environment type to use appropriate data readiness methods
                from config.env_loader import detect_environment
                env_type = detect_environment()

                if env_type.lower() == 'replit':
                    logger.info("Using Replit-specific storage for data readiness check")
                    # Initialize storage implementations for Replit
                    from config.storage import StorageFactory
                    storage = StorageFactory.create(config, env_type)

                    # Check if data is ready using appropriate storage implementations
                    complete_data_storage = storage['complete_data']
                    stratified_storage = storage['stratified_sample']

                    row_count = await complete_data_storage.get_row_count()
                    logger.info(f"PostgreSQL database has {row_count} rows")

                    stratified_exists = await stratified_storage.sample_exists()
                    logger.info(f"Stratified sample exists: {stratified_exists}")

                    # Force refresh if stratified sample doesn't exist
                    if not stratified_exists:
                        logger.info("Stratified sample doesn't exist in Replit storage, forcing refresh")
                        request.force_refresh = True

                    # Check if we need to prepare data
                    data_is_ready = (row_count > 0 and stratified_exists)
                    if not data_is_ready or request.force_refresh:
                        logger.info(f"Data not ready or force_refresh={request.force_refresh}, preparing data...")
                        await data_ops.ensure_data_ready(
                            force_refresh=request.force_refresh,
                            skip_embeddings=request.skip_embeddings
                        )
                else:
                    # Use standard file-based checks for Docker/local environment
                    if request.force_refresh or not await data_ops.is_data_ready(skip_embeddings=request.skip_embeddings):
                        logger.info(f"Data not ready or force_refresh={request.force_refresh}, preparing data...")
                        await data_ops.ensure_data_ready(
                            force_refresh=request.force_refresh,
                            skip_embeddings=request.skip_embeddings
                        )

                # Load the stratified data
                logger.info("Loading stratified data for processing...")
                library_df = await data_ops.load_stratified_data()

                # Log information about the data
                logger.info(f"Loaded stratified data with {len(library_df)} rows and columns: {list(library_df.columns)}")

                # Verify embeddings are present
                if 'embedding' not in library_df.columns:
                    logger.warning("Embeddings not present in loaded data, checking if they need to be loaded separately")
                    # Check if embeddings need to be loaded separately
                    try:
                        # Use the StorageFactory instead of direct attribute access on data_ops
                        from config.storage import StorageFactory
                        from config.env_loader import detect_environment
                        env_type = detect_environment()
                        storage = StorageFactory.create(data_ops.config, env_type)
                        embedding_storage = storage['embeddings']

                        embeddings, thread_map = await embedding_storage.get_embeddings()
                        if embeddings is not None and thread_map is not None:
                            logger.info(f"Merging {len(embeddings)} embeddings with stratified data...")
                            library_df["embedding"] = None

                            # Add embeddings to the DataFrame
                            matched = 0
                            for idx, row in library_df.iterrows():
                                thread_id = str(row["thread_id"])
                                if thread_id in thread_map:
                                    emb_idx = thread_map[thread_id]
                                    if isinstance(emb_idx, (int, str)) and str(emb_idx).isdigit():
                                        emb_idx = int(emb_idx)
                                        if 0 <= emb_idx < len(embeddings):
                                            library_df.at[idx, "embedding"] = embeddings[emb_idx]
                                            matched += 1

                            logger.info(f"Successfully matched {matched} embeddings out of {len(library_df)} rows")
                        else:
                            logger.warning("Failed to load separate embeddings, proceeding with potentially limited functionality")
                    except Exception as e:
                        logger.error(f"Error loading embeddings: {e}")
                        logger.error(traceback.format_exc())
                        logger.warning("Failed to load separate embeddings, proceeding with potentially limited functionality")

                # Ensure we have the necessary text field for inference
                if "text_clean" not in library_df.columns and "content" in library_df.columns:
                    logger.info("Adding text_clean field from content field")
                    library_df["text_clean"] = library_df["content"]

                # Now process the query with the loaded data
                logger.info(f"Processing query: '{request.query[:50]}...'")
                processing_start = time.time()

                result = await process_query(
                    query=request.query,
                    agent=agent,
                    library_df=library_df,
                    config=config,
                    character_slug=request.character_slug
                )

                processing_time = round((time.time() - processing_start) * 1000, 2)
                logger.info(f"Query processed in {processing_time}ms with {len(result.get('chunks', []))} chunks")

                # Store result
                success = await _store_batch_result(
                    batch_id=task_id,
                    result=result,
                    config=config
                )

                if not success:
                    raise ProcessingError(
                        message="Failed to store query results",
                        operation="store_result"
                    )

                # Update task status
                async with _tasks_lock:
                    if task_id in _background_tasks:
                        _background_tasks[task_id]["status"] = "completed"
                        _background_tasks[task_id]["progress"] = 100

                # Create complete response
                response = {
                    "status": "completed",
                    "task_id": task_id,
                    "chunks": result.get("chunks", []),
                    "summary": result.get("summary", ""),
                    "metadata": result.get("metadata", {})
                }

                duration_ms = round((time.time() - start_time) * 1000, 2)
                response["metadata"]["processing_time_ms"] = duration_ms

                # Save complete response
                try:
                    base_path = Path(Config.get_paths()["generated_data"])
                    json_path, embeddings_path = await save_query_output(
                        response=response,
                        base_path=base_path,
                        logger=logger,
                        query=request.query,
                        task_id=task_id
                    )

                    # Add file paths to response metadata
                    if "metadata" not in response:
                        response["metadata"] = {}
                    
                    # Handle both filesystem paths and object storage keys
                    response["metadata"]["saved_files"] = {}
                    
                    if json_path:
                        response["metadata"]["saved_files"]["json"] = str(json_path)
                    
                    if embeddings_path:
                        response["metadata"]["saved_files"]["embeddings"] = str(embeddings_path)
                    
                    # Add storage type to metadata
                    from config.env_loader import is_replit_environment
                    response["metadata"]["storage_type"] = "object_storage" if is_replit_environment() else "filesystem"

                except Exception as e:
                    logger.error(f"Error saving query output: {e}")
                    # Continue processing even if saving fails

                logger.info(f"Query {task_id} processed in {duration_ms}ms")
                return response

            except Exception as e:
                logger.error(f"Error processing query {task_id}: {str(e)}")
                logger.error(traceback.format_exc())

                # Update task status to failed
                async with _tasks_lock:
                    if task_id in _background_tasks:
                        _background_tasks[task_id]["status"] = "failed"
                        _background_tasks[task_id]["error"] = str(e)

                # Store error result
                error_result = {
                    "chunks": [],
                    "summary": f"Error processing query: {str(e)}",
                    "metadata": {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                }

                await _store_batch_result(
                    batch_id=task_id,
                    result=error_result,
                    config=config
                )

    except ValidationError as e:
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in base_query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "task_id": task_id})
    finally:
        # Log endpoint call
        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_endpoint_call(
            logger=logger,
            endpoint="/query",
            method="POST",
            duration_ms=duration_ms,
            params={"query": request.query[:50] + "..." if len(request.query) > 50 else request.query}
        )

async def _process_single_query(
    task_id: str,
    query: str,
    agent: KnowledgeAgent,
    config: ModelConfig,
    use_batching: bool = True,
    data_ops: Optional[DataOperations] = None,
    force_refresh: bool = False,
    skip_embeddings: bool = False,
    character_slug: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single query asynchronously."""
    logger.info(f"Processing query {task_id}: {query[:50]}...")

    try:
        # Update task status
        async with _tasks_lock:
            _background_tasks[task_id] = {
                "status": "processing",
                "timestamp": time.time(),
                "query": query,
                "progress": 10
            }

        # Check environment type to determine storage approach
        from config.env_loader import detect_environment
        env_type = detect_environment()

        # Get data operations if not provided
        if data_ops is None:
            data_ops = await get_data_ops()

        # Ensure data is ready based on environment
        if env_type.lower() == 'replit':
            logger.info("Using Replit-specific storage for data readiness check")
            # Initialize storage implementations for Replit
            from config.storage import StorageFactory
            storage = StorageFactory.create(data_ops.config, env_type)

            # Check if data preparation is needed
            complete_data_storage = storage['complete_data']
            stratified_storage = storage['stratified_sample']

            row_count = await complete_data_storage.get_row_count()
            logger.info(f"PostgreSQL database has {row_count} rows")

            stratified_exists = await stratified_storage.sample_exists()
            logger.info(f"Stratified sample exists: {stratified_exists}")

            # Force refresh if stratified sample doesn't exist
            if not stratified_exists:
                logger.info("Stratified sample doesn't exist in Replit storage, forcing refresh")
                force_refresh = True

            # Check if we need to prepare data
            data_is_ready = (row_count > 0 and stratified_exists)
            if not data_is_ready or force_refresh:
                logger.info(f"Data not ready or force_refresh={force_refresh}, preparing data...")
                await data_ops.ensure_data_ready(
                    force_refresh=force_refresh,
                    skip_embeddings=skip_embeddings
                )
        else:
            # Use standard file-based checks for Docker/local environment
            if force_refresh or not await data_ops.is_data_ready(skip_embeddings=skip_embeddings):
                logger.info(f"Data not ready or force_refresh={force_refresh}, preparing data...")
                await data_ops.ensure_data_ready(
                    force_refresh=force_refresh,
                    skip_embeddings=skip_embeddings
                )

        # Load stratified data for background processing
        logger.info("Loading stratified data for background processing...")
        library_df = await data_ops.load_stratified_data()
        logger.info(f"Loaded {len(library_df)} rows from stratified data")

        # Update task status
        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["progress"] = 30

        # Process query efficiently (with batching if enabled)
        if use_batching:
            # Use efficient batched processing
            start_time = time.time()
            result = await process_query(
                query=query,
                agent=agent,
                library_df=library_df,
                config=config,
                character_slug=character_slug
            )
            processing_time = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Query processed in {processing_time}ms with batching")
        else:
            # Use standard processing
            result = await agent.process_query(
                query=query,
                df=library_df,
                model=config.get_provider(ModelOperation.SUMMARIZATION)
            )

        # Update task status
        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["progress"] = 80

        # Store result for retrieval
        success = await _store_batch_result(
            batch_id=task_id,
            result=result,
            config=config
        )

        if not success:
            logger.warning(f"Failed to store results for task {task_id}")

        # Update task status
        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["status"] = "completed"
                _background_tasks[task_id]["progress"] = 100
                _background_tasks[task_id]["completed_at"] = time.time()

        # _store_batch_result now handles saving to disk, no need for separate save_query_output call

        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error processing query {task_id}: {str(e)}")
        logger.error(traceback.format_exc())

        # Update task status to failed
        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["status"] = "failed"
                _background_tasks[task_id]["error"] = str(e)
                _background_tasks[task_id]["progress"] = 100
                _background_tasks[task_id]["completed_at"] = time.time()

        # Store error result
        error_result = {
            "chunks": [],
            "summary": f"Error processing query: {str(e)}",
            "metadata": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

        await _store_batch_result(
            batch_id=task_id,
            result=error_result,
            config=config
        )

        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }

@router.post("/batch_process")
async def batch_process_queries(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """Process multiple queries in an optimized batch."""
    start_time = time.time()
    batch_id = _generate_task_id(prefix="batch")
    logger.info(f"Processing batch of {len(request.queries)} queries: {batch_id}")

    try:
        # Initialize the agent and config
        agent = await get_agent()
        config = load_config()

        # Initialize data operations
        data_ops = DataOperations(DataConfig.from_config())

        # Prepare data using the Chanscope approach
        # This ensures stratified data is available before processing
        stratified_data = await _prepare_data_if_needed(
            config=config,
            data_ops=data_ops,
            force_refresh=request.force_refresh,
            skip_embeddings=request.skip_embeddings
        )

        if stratified_data is None or stratified_data.empty:
            raise ProcessingError(
                message="Failed to prepare stratified data for batch processing",
                operation="batch_process"
            )

        logger.info(f"Using stratified data with {len(stratified_data)} records for batch processing")

        results = await process_multiple_queries_efficient(
            queries=request.queries,
            agent=agent,
            stratified_data=stratified_data,  # Pass the prepared stratified data directly
            chunk_batch_size=request.chunk_batch_size or config.chunk_batch_size,
            summary_batch_size=request.summary_batch_size or config.summary_batch_size,
            max_workers=request.max_workers or config.max_workers,
            providers={
                ModelOperation.EMBEDDING: config.get_provider(ModelOperation.EMBEDDING),
                ModelOperation.CHUNK_GENERATION: config.get_provider(ModelOperation.CHUNK_GENERATION),
                ModelOperation.SUMMARIZATION: config.get_provider(ModelOperation.SUMMARIZATION)
            },
            character_slug=request.character_slug
        )

        # Log processing time
        duration_ms = round((time.time() - start_time) * 1000, 2)
        avg_time_per_query = round(duration_ms / len(request.queries), 2)
        logger.info(f"Batch processed {len(request.queries)} queries in {duration_ms}ms (avg: {avg_time_per_query}ms/query)")

        # Save individual results with unique IDs
        saved_result_info = []
        for i, (query, result) in enumerate(zip(request.queries, results)):
            # Generate a unique ID for each result in the batch
            result_id = f"{batch_id}_item_{i}"
            
            # Create a config-like object with the query for better filenaming
            result_config = copy.deepcopy(config)
            result_config.query = query
            
            # Create a structured result dictionary
            result_dict = {
                "chunks": result[0],  # First element is chunks
                "summary": result[1],  # Second element is summary
                "metadata": {
                    "batch_id": batch_id,
                    "item_index": i,
                    "query": query,
                    "processing_time_ms": duration_ms / len(request.queries)  # Approximate per-query time
                }
            }
            
            # Store the result both in memory and on disk
            success = await _store_batch_result(
                batch_id=result_id,
                result=result_dict,
                config=result_config,
                save_to_disk=True
            )
            
            if success:
                # Track successful saves
                saved_info = {
                    "result_id": result_id,
                    "query": query,
                    "saved": success
                }
                if result_id in _batch_results and "metadata" in _batch_results[result_id]:
                    # Add saved file paths if available
                    if "saved_files" in _batch_results[result_id]["metadata"]:
                        saved_info["file_paths"] = _batch_results[result_id]["metadata"]["saved_files"]
                
                saved_result_info.append(saved_info)

        # Return results with metadata including save information
        return {
            "batch_id": batch_id,
            "results": results,
            "metadata": {
                "total_time_ms": duration_ms,
                "avg_time_per_query_ms": avg_time_per_query,
                "queries_processed": len(results),
                "saved_results": saved_result_info,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "batch_id": batch_id,
            "results": [],
            "error": str(e),
            "metadata": {
                "total_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": datetime.now().isoformat()
            }
        }

@router.get("/batch_status/{task_id}")
async def get_batch_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a processing task.

    This endpoint works with both query tasks and batch processing tasks.
    The task_id format is prefix_timestamp_randomhex, where prefix can be
    'query', 'batch', or other task types.
    """
    start_time = time.time()
    try:
        logger.debug(f"get_batch_status: Checking status for task_id: {task_id}")
        logger.debug(f"_background_tasks keys: {list(_background_tasks.keys())}")
        logger.debug(f"_batch_results keys: {list(_batch_results.keys())}")
        logger.debug(f"_query_batch length: {len(_query_batch)}")

        # First, check background tasks
        if task_id in _background_tasks:
            task_status = _background_tasks[task_id]
            logger.debug(f"Task {task_id} found in _background_tasks: {task_status}")
            response = {
                "status": task_status["status"],
                "total": task_status.get("total_queries"),
                "completed": task_status.get("completed_queries"),
                "results": task_status.get("results", []),
                "errors": task_status.get("errors", []),
                "duration_ms": task_status.get("duration_ms")
            }
        # Then check results cache
        elif task_id in _batch_results:
            result = _batch_results[task_id]
            logger.debug(f"Task {task_id} found in _batch_results: {result}")
            response = {
                "status": "completed",
                "result": result
            }
        else:
            # Check if the query is still in the queue
            position = None
            for i, item in enumerate(_query_batch):
                if item["id"] == task_id:
                    position = i
                    break
            if position is not None:
                logger.debug(f"Task {task_id} found in _query_batch at position {position}")
                response = {
                    "status": "queued",
                    "position": position,
                    "eta_seconds": _estimate_processing_time()
                }
            else:
                # Fallback: check the cache
                cache_key = f"task_result:{task_id}"
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Task {task_id} retrieved from cache.")
                    response = {
                        "status": "completed",
                        "result": cached_result
                    }
                else:
                    # Attempt to parse task_id if it follows the expected format
                    try:
                        parts = task_id.split("_")
                        if len(parts) < 3:
                            raise ValueError("Invalid task id format.")

                        prefix = parts[0]
                        timestamp_str = parts[1]
                        timestamp = int(timestamp_str)
                        current_time = int(time.time())

                        logger.debug(f"Extracted timestamp {timestamp} from task_id {task_id}; current_time={current_time}")

                        if current_time - timestamp < 600:
                            # Check if embedding generation is active
                            is_embedding_active = False
                            async with _tasks_lock:
                                if _embedding_task_key in _background_tasks and _background_tasks[_embedding_task_key]["status"] == "running":
                                    is_embedding_active = True
                            if is_embedding_active:
                                logger.debug(f"Embedding generation is active for task {task_id}")
                                response = {
                                    "status": "preprocessing",
                                    "message": "Embeddings are currently being generated. Please try again shortly.",
                                    "estimated_wait_time": 300  # 5 minutes
                                }
                            else:
                                logger.debug(f"Task {task_id} is processing but not found in any storage; returning generic processing message.")
                                response = {
                                    "status": "processing",
                                    "message": "The task is being processed. Please check back soon.",
                                    "task_id": task_id,
                                    "created_at": datetime.fromtimestamp(timestamp, tz=pytz.UTC).isoformat()
                                }
                        else:
                            logger.debug(f"Task {task_id} has expired based on timestamp check.")
                            # Check persistent history if available
                            history_path = Path(Config.get_paths()['logs']) / "batch_history.json"
                            if history_path.exists():
                                try:
                                    with open(history_path, 'r') as f:
                                        history = json.load(f)
                                    if task_id in history:
                                        task_info = history[task_id]
                                        if task_info.get("status") == "completed":
                                            response = {
                                                "status": "expired",
                                                "message": f"Task {task_id} was completed but results have expired. Results are only kept for 10 minutes.",
                                                "completed_at": task_info.get("completed_at"),
                                                "task_id": task_id
                                            }
                                            return response
                                        elif task_info.get("status") == "failed":
                                            response = {
                                                "status": "failed",
                                                "message": f"Task {task_id} failed: {task_info.get('error', 'Unknown error')}",
                                                "task_id": task_id,
                                                "error": task_info.get("error")
                                            }
                                            return response
                                except Exception as ex:
                                    logger.warning(f"Error reading batch history for {task_id}: {ex}")
                            raise ValidationError(
                                message=f"Task {task_id} not found",
                                field="task_id",
                                value=task_id
                            )
                    except (IndexError, ValueError) as ex:
                        raise ValidationError(
                            message=f"Invalid task ID format: {task_id}. Error: {ex}",
                            field="task_id",
                            value=task_id
                        )
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.debug(f"get_batch_status for {task_id} took {duration_ms} ms with response: {response}")
        return response

    except ValidationError as e:
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
        error = ProcessingError(
            message="Error retrieving batch status",
            operation="get_batch_status",
            resource=task_id,
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

@router.get("/process_recent_query")
async def process_recent_query(
    background_tasks: BackgroundTasks,
    agent: KnowledgeAgent = Depends(get_agent),
    data_ops: DataOperations = Depends(get_data_ops),
    select_board: Optional[str] = None,
    task_id: Optional[str] = None,
    use_background: bool = Query(False, title="Run in background", description="Process the query in the background"),
    filter_date: Optional[str] = None,
    force_refresh: bool = Query(False, title="Force refresh", description="Force data refresh")
):
    """Process a query using recent data from the last 6 hours with batch processing.

    This endpoint aligns with process_query by using the same batch processing system,
    ensuring consistent behavior and response formats across both endpoints.

    Args:
        background_tasks: For background processing
        agent: Knowledge agent for processing 
        data_ops: Data operations for preparation
        select_board: Optional board ID to filter data
        task_id: Optional user-provided task ID
        use_background: Whether to process the query in the background
        filter_date: Optional date filter to override the default 6-hour window
        force_refresh: Whether to force data refresh
    """
    try:
        # Check environment type to determine if we need to force refresh data
        from config.env_loader import detect_environment
        env_type = detect_environment()

        if env_type.lower() == 'replit':
            # Check if stratified sample exists in Replit storage
            from config.storage import StorageFactory
            storage = StorageFactory.create(data_ops.config, env_type)
            stratified_storage = storage['stratified_sample']

            stratified_exists = await stratified_storage.sample_exists()
            if not stratified_exists:
                logger.info("Stratified sample doesn't exist in Replit storage, forcing refresh")
                force_refresh = True

        # Calculate time range
        end_time = datetime.now(pytz.UTC)

        # Always prioritize 6-hour lookback for recency
        try:
            # Default behavior: always look back 6 hours for current data
            start_time = end_time - timedelta(hours=12)
            logger.info(f"Using default 6-hour window: {start_time.isoformat()} to {end_time.isoformat()}")
        except Exception as e:
            # Fallback to filter_date only if there's an issue with the 6-hour calculation
            logger.warning(f"Error calculating 6-hour window: {e}. Attempting to use filter_date instead.")
            if filter_date:
                try:
                    start_time = pd.to_datetime(filter_date, utc=True)
                    logger.info(f"Using provided filter_date as fallback: {filter_date}")
                except Exception as e2:
                    logger.error(f"Invalid filter_date format: {filter_date}. Error: {e2}. Using 24-hour emergency fallback.")
                    start_time = end_time - timedelta(hours=24)
            else:
                logger.error("Unable to calculate window and no filter_date provided. Using 24-hour emergency fallback.")
                start_time = end_time - timedelta(hours=24)

        query_filter_date = start_time.strftime('%Y-%m-%d %H:%M:%S+00:00')

        # Get base settings and paths
        base_settings = get_base_settings()
        paths = base_settings.get('paths', {})
        stored_queries_path = Path(paths.get('config', 'config')) / 'stored_queries.yaml'

        # Load stored query based on environment
        if env_type.lower() == 'replit':
            logger.info("Using Replit environment, loading query from configuration")
            # In Replit, we may not have direct file access, use a default query if file can't be accessed
            try:
                with open(stored_queries_path, 'r') as f:
                    stored_queries = yaml.safe_load(f)
                
                # Get random query based on board selection
                if stored_queries and 'query' in stored_queries:
                    import random
                    
                    if select_board == "biz" and 'biz' in stored_queries['query'] and 'queries' in stored_queries['query']['biz']:
                        # Get random query from biz board queries
                        biz_queries = stored_queries['query']['biz']['queries']
                        if biz_queries:
                            query = random.choice(biz_queries)
                            logger.info(f"Using random biz query: {query}")
                        else:
                            # Fallback for empty biz queries
                            query = "Stocks, Defi, financial market impact and cryptocurrency developments <Bitcoin, Ethereum, Solana, Chainlink, Base, MSTR, NASDAQ, S&P 500, Dow Jones, and other financial markets>"
                            logger.info(f"Using default biz query (no queries found): {query}")
                    
                    elif select_board == "pol" and 'pol' in stored_queries['query'] and 'queries' in stored_queries['query']['pol']:
                        # Get random query from pol board queries
                        pol_queries = stored_queries['query']['pol']['queries']
                        if pol_queries:
                            query = random.choice(pol_queries)
                            logger.info(f"Using random pol query: {query}")
                        else:
                            # Fallback for empty pol queries
                            query = "Current events, election results, executive orders, legislation, geopolitical developments, and regulatory developments"
                            logger.info(f"Using default pol query (no queries found): {query}")
                    
                    else:
                        # No specific board or invalid board, combine all queries and select random one
                        all_queries = []
                        if 'biz' in stored_queries['query'] and 'queries' in stored_queries['query']['biz']:
                            all_queries.extend(stored_queries['query']['biz']['queries'])
                        if 'pol' in stored_queries['query'] and 'queries' in stored_queries['query']['pol']:
                            all_queries.extend(stored_queries['query']['pol']['queries'])
                        
                        if all_queries:
                            query = random.choice(all_queries)
                            logger.info(f"Using random query (no board specified): {query}")
                        else:
                            # Fallback if no queries found in any board
                            query = "Current geopolitical events and financial market developments"
                            logger.info(f"Using default query (no queries found): {query}")
                else:
                    # Fallback for malformed stored_queries yaml
                    if select_board == "biz":
                        query = "Stocks, Defi, financial market impact and cryptocurrency developments <Bitcoin, Ethereum, Solana, Chainlink, Base, MSTR, NASDAQ, S&P 500, Dow Jones, and other financial markets>"
                    elif select_board == "pol":
                        query = "Current events, election results, executive orders, legislation, geopolitical developments, and regulatory developments"
                    else:
                        query = "Current geopolitical events and financial market developments"
                    logger.info(f"Using default query (malformed stored_queries): {query}")
            except FileNotFoundError:
                # Use default query if file doesn't exist
                query = "Current events with financial market impact and cryptocurrency developments"
                logger.info(f"Using default query (file not found): {query}")
            except Exception as e:
                logger.warning(f"Error loading stored queries: {e}. Using default query.")
                query = "Current events with financial market impact and cryptocurrency developments"
        else:
            # In Docker/local environment, try to load from file
            try:
                with open(stored_queries_path, 'r') as f:
                    stored_queries = yaml.safe_load(f)
                if not stored_queries or 'query' not in stored_queries or 'example' not in stored_queries['query']:
                    raise ValidationError(
                        message="No example queries found in stored_queries.yaml",
                        field="stored_queries",
                        value=None
                    )
                query = stored_queries['query']['example'][0]
            except FileNotFoundError as e:
                raise ConfigurationError(
                    message="Stored queries file not found",
                    config_key="stored_queries_path",
                    config_value=str(stored_queries_path))
            except yaml.YAMLError as e:
                raise ConfigurationError(
                    message="Invalid YAML format in stored queries",
                    config_key="stored_queries",
                    original_error=e)

        # Create request object with consistent parameters and pass task_id
        request = QueryRequest(
            query=query,
            filter_date=query_filter_date,
            force_refresh=force_refresh,
            skip_embeddings=True,
            skip_batching=False,
            select_board=select_board,
            task_id=task_id,
            use_background=use_background,
            character_slug=None  # Default to None as this is a system-generated query
        )

        # Log request
        log_params = {
            "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "filter_date": query_filter_date,
            "select_board": select_board,
            "use_background": use_background,
            "force_refresh": force_refresh
        }
        log_endpoint_call(
            logger=logger,
            endpoint="/process_recent_query",
            method="GET",
            duration_ms=None,
            params=log_params
        )

        # Reuse process_query endpoint logic for consistency
        result = await base_query(
            request=request,
            background_tasks=background_tasks)

        # Add time range information while maintaining batch processing response format
        if isinstance(result, dict) and "status" in result:
            result["time_range"] = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }

        return result

    except ValidationError as e:
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except ConfigurationError as e:
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
        error = ProcessingError(
            message="Error processing recent query",
            operation="process_recent_query",
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

async def _run_embedding_generation(data_ops: DataOperations) -> None:
    """Background task for generating embeddings.

    This function handles the embedding generation process with proper state tracking
    and error handling. It ensures that:
    1. Embeddings are only generated for new or modified data
    2. The process is properly logged and monitored
    3. Errors are caught and logged appropriately

    Args:
        data_ops: DataOperations instance for data processing
    """
    try:
        logger.info("Starting background embedding generation")
        async with _tasks_lock:
            _background_tasks[_embedding_task_key] = {
                'status': 'running',
                'start_time': datetime.now(pytz.UTC),
                'progress': 0
            }

        # Load existing stratified sample
        stratified_file = data_ops.config.stratified_data_path / 'stratified_sample.csv'
        if not stratified_file.exists():
            error_msg = "Stratified sample not found for embedding generation"
            logger.error(error_msg)
            async with _tasks_lock:
                _background_tasks[_embedding_task_key].update({
                    'status': 'failed',
                    'error': error_msg,
                    'end_time': datetime.now(pytz.UTC)
                })
            return

        df = pd.read_csv(stratified_file)
        if df.empty:
            error_msg = "Stratified sample is empty, cannot generate embeddings"
            logger.warning(error_msg)
            async with _tasks_lock:
                _background_tasks[_embedding_task_key].update({
                    'status': 'failed',
                    'error': error_msg,
                    'end_time': datetime.now(pytz.UTC)
                })
            return

        # Generate embeddings using existing stratified data
        total_rows = len(df)
        async with _tasks_lock:
            _background_tasks[_embedding_task_key]['total_rows'] = total_rows

        await data_ops._update_embeddings(
            stratified_data=df,
            progress_callback=lambda progress: _update_embedding_progress(progress, total_rows)
        )

        async with _tasks_lock:
            _background_tasks[_embedding_task_key].update({
                'status': 'completed',
                'end_time': datetime.now(pytz.UTC),
                'progress': 100
            })
        logger.info("Background embedding generation completed successfully")

    except Exception as e:
        error_msg = f"Background embedding generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        async with _tasks_lock:
            _background_tasks[_embedding_task_key].update({
                'status': 'failed',
                'error': error_msg,
                'end_time': datetime.now(pytz.UTC)
            })

async def _update_embedding_progress(current: int, total: int) -> None:
    """Update embedding generation progress in a thread-safe manner.

    Args:
        current: Current number of processed rows
        total: Total number of rows to process
    """
    async with _tasks_lock:
        if _embedding_task_key in _background_tasks:
            progress = min(100, int((current / total) * 100))
            _background_tasks[_embedding_task_key]['progress'] = progress
            logger.info(f"Embedding generation progress: {progress}%")

@router.post("/trigger_embedding_generation")
async def trigger_embedding_generation(
    background_tasks: BackgroundTasks,
    data_ops: DataOperations = Depends(get_data_ops)
):
    """Trigger background embedding generation process.

    This endpoint ensures that:
    1. Only one embedding generation process runs at a time
    2. The process state is properly tracked
    3. Errors are handled and reported appropriately

    Returns:
        Dict[str, str]: Status message indicating whether the process started
    """
    try:
        async with _tasks_lock:
            if (_embedding_task_key in _background_tasks and 
                _background_tasks[_embedding_task_key].get('status') == 'running'):
                return {
                    "status": "already_running",
                    "message": "Embedding generation is already in progress",
                    "task_id": _embedding_task_key
                }

        # Check if stratification is needed before proceeding
        if data_ops.needs_stratification():
            logger.warning("Stratified data needs to be updated before generating embeddings")
            return {
                "status": "needs_stratification",
                "message": "Stratified data needs to be updated before generating embeddings. Please trigger stratification first.",
                "task_id": None
            }

        # Create background task
        background_tasks.add_task(_run_embedding_generation, data_ops)

        return {
            "status": "started",
            "message": "Embedding generation started in background",
            "task_id": _embedding_task_key
        }
    except Exception as e:
        error = ProcessingError(
            message=f"Error triggering embedding generation: {str(e)}",
            operation="embedding_generation",
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

@router.get("/embedding_status")
async def get_embedding_status():
    """Get detailed status of background embedding generation.

    Returns:
        Dict[str, Any]: Detailed status information including:
        - Current status (not_started/running/completed/failed)
        - Progress percentage
        - Error message if failed
        - Start and end times
    """
    async with _tasks_lock:
        if _embedding_task_key not in _background_tasks:
            return {
                "status": "not_started",
                "message": "No embedding generation task has been started"
            }

        task_info = _background_tasks[_embedding_task_key]
        return {
            "status": task_info['status'],
            "message": (
                task_info.get('error', "Embedding generation completed successfully")
                if task_info['status'] in ['completed', 'failed']
                else "Embedding generation is in progress"
            ),
            "progress": task_info.get('progress', 0),
            "start_time": task_info['start_time'].isoformat(),
            "end_time": task_info.get('end_time', None),
            "total_rows": task_info.get('total_rows', 0)
        }

@router.get("/debug/routes")
async def debug_routes(request: Request):
    """Debug endpoint to list all registered routes."""
    routes = []
    for route in request.app.routes:
        methods = getattr(route, "methods", None)
        if methods:
            routes.append({
                "endpoint": route.name,
                "methods": sorted(list(methods)),
                "path": route.path,
                "is_api": route.path.startswith("/api/")
            })
    routes.sort(key=lambda x: (0 if x["is_api"] else 1, x["path"]))
    return {
        "routes": routes,
        "total_routes": len(routes),
        "api_routes": sum(1 for r in routes if r["is_api"]),
        "other_routes": sum(1 for r in routes if not r["is_api"])
    }

@router.api_route("/debug/request", methods=["GET", "POST", "OPTIONS"], operation_id="debug_request_unique")
async def debug_request(request: Request):
    """Debug endpoint to show request details."""
    try:
        details = {
            "method": request.method,
            "path": request.url.path,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params)
        }
        if request.method == "POST":
            try:
                details["json_body"] = await request.json()
            except Exception as e:
                details["json_error"] = str(e)
        logger.info(f"Debug request details: {details}")
        return details
    except Exception as e:
        logger.error(f"Error in debug request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/cache")
async def cache_health() -> Dict[str, Any]:
    """Health check for cache."""
    try:
        # Get cache metrics from simple counters
        # Handle both SimpleCounter class and global variable implementations
        if hasattr(CACHE_HITS, 'get_value'):
            hits = CACHE_HITS.get_value()
            misses = CACHE_MISSES.get_value()
            errors = CACHE_ERRORS.get_value()
        else:
            # Fall back to direct access for the simplified test implementation
            hits = CACHE_HITS
            misses = CACHE_MISSES
            errors = CACHE_ERRORS

        total_requests = hits + misses

        cache_status = {
            "status": "healthy",
            "type": "in_memory",
            "metrics": {
                "hit_ratio": f"{(hits/total_requests)*100:.2f}%" if total_requests > 0 else "0%",
                "hits": int(hits),
                "misses": int(misses),
                "errors": int(errors),
                "total_requests": int(total_requests)
            },
            "configuration": {
                "enabled": True,
                "ttl": 3600
            }
        }

        return cache_status
    except Exception as e:
        error = ProcessingError(
            message="Error checking cache health",
            operation="cache_health",
            resource="in_memory"
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

@router.get("/api_health")
async def api_health():
    """Simple health check endpoint for API health checks."""
    return {
        "status": "ok", 
        "ready": True
    }

@router.get("/health/embeddings", response_model=Dict[str, Any])
async def embedding_health(
    data_ops: DataOperations = Depends(get_data_ops)
) -> Dict[str, Any]:
    """Get detailed health metrics about embedding coverage and quality.

    Returns:
        Dict containing metrics about embedding coverage, dimensions, and quality
    """
    try:
        metrics = await data_ops.get_embedding_coverage_metrics()

        # Add additional health checks
        health_status = "healthy"
        issues = []

        # Check coverage
        if metrics["coverage_percentage"] < 90:
            health_status = "degraded"
            issues.append(f"Low embedding coverage: {metrics['coverage_percentage']:.1f}%")

        # Check dimension mismatches
        if metrics["dimension_mismatches"] > 0:
            health_status = "degraded"
            issues.append(f"Found {metrics['dimension_mismatches']} embeddings with incorrect dimensions")

        # Check if using mock data
        if metrics["is_mock_data"]:
            health_status = "degraded"
            issues.append("Using mock embeddings instead of real embeddings")

        return {
            "status": health_status,
            "issues": issues,
            "metrics": metrics,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }

    except Exception as e:
        logger.error(f"Error checking embedding health: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }

def handle_error(e: Exception) -> None:
    """Centralized error handler for converting exceptions to HTTPException."""
    if isinstance(e, APIError):
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    elif isinstance(e, ModelConfigurationError):
        error = ConfigurationError(
            message=str(e),
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )
    elif isinstance(e, ModelProviderError):
        error = ProcessingError(
            message=str(e),
            operation="model_provider",
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )
    elif isinstance(e, ModelOperationError):
        error = ProcessingError(
            message=str(e),
            operation="model_operation",
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )
    else:
        error = APIError(
            message="Internal server error",
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            details={"error": str(e)},
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

async def _add_to_query_batch(request: QueryRequest) -> str:
    """Add query to batch queue and return batch ID."""
    batch_id = _generate_task_id(prefix="batch")
    _query_batch.append({
        "id": batch_id,
        "request": request,
        "timestamp": datetime.now(pytz.UTC)})
    return batch_id

def _should_process_batch() -> bool:
    """Determine if batch should be processed based on size and wait time."""
    if len(_query_batch) >= _BATCH_SIZE:
        return True
    if _query_batch and (datetime.now(pytz.UTC) - _query_batch[0]["timestamp"]).total_seconds() >= _BATCH_WAIT_TIME:
        return True
    return False

def _estimate_processing_time() -> int:
    """Estimate processing time based on batch size and historical data."""
    batch_size = len(_query_batch)
    avg_time_per_query = _last_batch_processing_time / _BATCH_SIZE if _last_batch_processing_time > 0 else 2
    position_in_queue = len(_query_batch) - 1
    return int(avg_time_per_query * (position_in_queue // _BATCH_SIZE + 1))

async def _prepare_data_if_needed(
    config: ModelConfig,
    data_ops: DataOperations,
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> pd.DataFrame:
    """
    Ensures data is ready and loads the stratified dataset.

    Args:
        config: Model configuration
        data_ops: Data operations instance
        force_refresh: Whether to force data refresh
        skip_embeddings: Whether to skip embedding generation

    Returns:
        Stratified data with embeddings
    """
    # Check environment type to use appropriate data readiness methods
    from config.env_loader import detect_environment
    env_type = detect_environment()
    logger.info(f"Preparing data in {env_type} environment (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})")

    # Step 1: Check if data is ready (using environment-appropriate methods)
    if env_type.lower() == 'replit':
        logger.info("Using Replit-specific storage for data readiness check")
        # Initialize storage implementations for Replit
        from config.storage import StorageFactory
        storage = StorageFactory.create(data_ops.config, env_type)

        # Check if data is ready using appropriate storage implementations
        complete_data_storage = storage['complete_data']
        stratified_storage = storage['stratified_sample']

        row_count = await complete_data_storage.get_row_count()
        logger.info(f"PostgreSQL database has {row_count} rows")

        stratified_exists = await stratified_storage.sample_exists()
        logger.info(f"Stratified sample exists: {stratified_exists}")

        data_is_ready = (row_count > 0 and stratified_exists)
        if not data_is_ready or force_refresh:
            logger.info(f"Data not ready or force_refresh={force_refresh}, preparing data...")
            success = await data_ops.ensure_data_ready(
                force_refresh=force_refresh,
                skip_embeddings=skip_embeddings
            )
            if not success:
                logger.error("Failed to prepare data")
                raise HTTPException(
                    status_code=500,
                    detail={"error": "Failed to prepare data", "status": "error"}
                )
    else:
        # Use standard file-based checks for Docker/local environment
        is_ready = await data_ops.is_data_ready(skip_embeddings=skip_embeddings)
        if not is_ready or force_refresh:
            logger.info(f"Data not ready or force_refresh={force_refresh}, preparing data...")
            success = await data_ops.ensure_data_ready(
                force_refresh=force_refresh,
                skip_embeddings=skip_embeddings
            )
            if not success:
                logger.error("Failed to prepare data")
                raise HTTPException(
                    status_code=500,
                    detail={"error": "Failed to prepare data", "status": "error"}
                )

    # Step 2: Load stratified data
    logger.info("Loading stratified data...")
    try:
        # Use data_ops method to load stratified data (handles both environments)
        stratified_data = await data_ops.load_stratified_data()
        logger.info(f"Loaded stratified data with {len(stratified_data)} rows")

        # Ensure we have the necessary columns
        required_columns = ['thread_id', 'text_clean']
        missing_columns = [col for col in required_columns if col not in stratified_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"Missing required columns in stratified data: {missing_columns}",
                    "status": "error"
                }
            )

        return stratified_data
    except Exception as e:
        logger.error(f"Error loading stratified data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": f"Error loading stratified data: {str(e)}", "status": "error"}
        )

async def _store_batch_result(batch_id: str, result: Union[tuple, Dict[str, Any]], config: Union[Dict[str, Any], ModelConfig], save_to_disk: bool = True) -> bool:
    """Store batch processing results for later retrieval and optionally save to disk.

    Args:
        batch_id: Unique identifier for the task (format: prefix_timestamp_randomhex)
        result: Either a tuple of (chunks, summary) or a dict with chunks, summary, and metadata
        config: Either a ModelConfig object or a dictionary with configuration
        save_to_disk: Whether to save the result to disk in generated_data directory
        
    Returns:
        bool: Whether the operation was successful
    """
    # Not active, so set to None
    redis_client = None 
    try:
        # Handle both tuple and dict formats for backward compatibility
        if isinstance(result, tuple):
            if len(result) == 2:
                chunks, summary = result
                metadata = {}
            elif len(result) == 3:
                chunks, summary, metadata = result
            else:
                raise ValueError(f"Invalid result tuple length: {len(result)}")
        else:
            # Extract from dictionary format
            chunks = result.get("chunks", [])
            summary = result.get("summary", "")
            metadata = result.get("metadata", {}) or {}

        # Get batch_result_ttl from config
        if isinstance(config, ModelConfig):
            batch_result_ttl = getattr(config, 'batch_result_ttl', 300)  # Default 5 minutes
        else:
            # If config is a dict, get ttl from it or use default
            batch_result_ttl = config.get('batch_result_ttl', 300)

        # Create a result object
        result_obj = {
            "id": batch_id,
            "status": "completed",
            "chunks": chunks,
            "summary": summary,
            "metadata": {
                "timestamp": time.time(),
                "expiry": time.time() + batch_result_ttl,
                "processing_time_ms": metadata.get("processing_time_ms"),
                "num_chunks": len(chunks),
                "num_relevant_strings": metadata.get("num_relevant_strings"),
                "temporal_context": metadata.get("temporal_context"),
                "batch_sizes": metadata.get("batch_sizes"),
                "batching_enabled": metadata.get("batching_enabled", True)
            }
        }

        logger.debug(f"Storing result for task {batch_id}")

        # Store in Redis or another shared storage
        if redis_client:
            redis_client.set(f"task_result:{batch_id}", json.dumps({
                'result': result_obj,
                'expires_at': time.time() + batch_result_ttl,
                'config': config if isinstance(config, dict) else config.dict()
            }), ex=batch_result_ttl)
        else:
            logger.debug("Redis client not available, skipping result storage")

        # Update batch history
        await _update_batch_history({batch_id: {'timestamp': time.time(), 'query': config.query if hasattr(config, 'query') else ''}})

        logger.info(f"Stored result for task {batch_id} (expires in {batch_result_ttl}s)")

        # Also store results in background tasks for redundancy
        if batch_id in _background_tasks:
            _background_tasks[batch_id]['results'] = result

        # Store in memory cache
        _batch_results[batch_id] = result_obj
        
        # Save result to disk if requested
        if save_to_disk:
            try:
                # Extract query from config if available for better filenaming
                query = config.query if hasattr(config, 'query') else None
                
                # Use save_query_output from knowledge_agents.utils to save result to disk
                from knowledge_agents.utils import save_query_output
                
                # Create a full response object that matches what save_query_output expects
                response_for_saving = {
                    "task_id": batch_id,
                    "chunks": chunks,
                    "summary": summary,
                    "metadata": result_obj.get("metadata", {}).copy()
                }
                
                # Add original query to metadata if available
                if query:
                    response_for_saving["query"] = query
                
                # Get base path from Config
                base_path = Path(Config.get_paths().get('generated_data', 'data/generated_data'))
                
                # Call save_query_output to save result to disk
                json_path, embeddings_path = await save_query_output(
                    response=response_for_saving,
                    base_path=base_path,
                    task_id=batch_id,
                    query=query,
                    logger=logger
                )
                
                # Add saved file paths to metadata
                if json_path or embeddings_path:
                    saved_files = {}
                    if json_path:
                        saved_files["json"] = str(json_path)
                    if embeddings_path:
                        saved_files["embeddings"] = str(embeddings_path)
                    
                    # Update memory cache with file paths
                    if "metadata" not in _batch_results[batch_id]:
                        _batch_results[batch_id]["metadata"] = {}
                    _batch_results[batch_id]["metadata"]["saved_files"] = saved_files
                    
                    logger.info(f"Saved task {batch_id} results to disk: {json_path}")
                
            except Exception as save_error:
                logger.warning(f"Could not save results to disk for task {batch_id}: {save_error}")
                # Continue even if saving fails - don't affect the overall operation

        return True

    except Exception as e:
        logger.error(f"Error storing result: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def _cleanup_old_results():
    """Clean up old results to prevent memory leaks and disk space issues."""
    try:
        # Memory cleanup settings
        retention_period = 600  # 10 minutes
        current_time = time.time()
        batch_ids_to_remove = []
        batch_history_updates = {}
        disk_files_cleaned = 0
        
        # First perform memory cleanup
        # Check all stored results
        for batch_id, result in _batch_results.items():
            # Check if the result has expired
            timestamp = result.get("metadata", {}).get("timestamp", 0)
            if current_time - timestamp > retention_period:
                batch_ids_to_remove.append(batch_id)
                # Update history with expiry information
                batch_history_updates[batch_id] = {
                    "status": "expired",
                    "expired_at": current_time
                }

        # Remove expired results from memory
        if batch_ids_to_remove:
            memory_freed = 0
            for batch_id in batch_ids_to_remove:
                if batch_id in _batch_results:
                    # Estimate memory usage
                    try:
                        result_size = len(str(_batch_results[batch_id]))
                        memory_freed += result_size
                    except:
                        pass

                    # Remove from memory
                    del _batch_results[batch_id]

                    # Also try to remove from the cache
                    try:
                        cache_key = f"task_result:{batch_id}"
                        asyncio.create_task(cache.delete(cache_key))
                    except Exception as cache_error:
                        logger.debug(f"Cache removal error for {batch_id}: {str(cache_error)}")

            # Update batch history
            await _update_batch_history(batch_history_updates)

            logger.info(f"Cleaned up {len(batch_ids_to_remove)} old results from memory (retention: {retention_period}s)")
            if memory_freed > 0:
                logger.info(f"Estimated memory freed: {memory_freed / 1024:.2f} KB")

        # Now perform disk cleanup for generated_data directory
        try:
            # Use a longer retention period for disk files (1 day = 86400 seconds)
            disk_retention_period = 86400  # 1 day
            
            # Get base path for generated data
            base_path = Path(Config.get_paths().get('generated_data', 'data/generated_data'))
            if base_path.exists():
                # Look for JSON files older than retention period
                for json_file in base_path.glob("**/*.json"):
                    try:
                        # Get file modification time
                        file_mtime = json_file.stat().st_mtime
                        file_age = current_time - file_mtime
                        
                        # Delete files older than retention period
                        if file_age > disk_retention_period:
                            # Check for corresponding embedding files
                            embedding_json = base_path / "embeddings" / f"{json_file.stem}_embeddings.json"
                            embedding_npz = base_path / "embeddings" / f"{json_file.stem}_embeddings.npz"
                            
                            # Delete JSON file
                            json_file.unlink()
                            disk_files_cleaned += 1
                            
                            # Delete embedding files if they exist
                            if embedding_json.exists():
                                embedding_json.unlink()
                                disk_files_cleaned += 1
                            if embedding_npz.exists():
                                embedding_npz.unlink()
                                disk_files_cleaned += 1
                    except Exception as file_error:
                        logger.warning(f"Error cleaning up file {json_file}: {str(file_error)}")
                
                if disk_files_cleaned > 0:
                    logger.info(f"Cleaned up {disk_files_cleaned} old files from disk (retention: {disk_retention_period/3600:.1f} hours)")
        except Exception as disk_error:
            logger.error(f"Error during disk cleanup: {str(disk_error)}")
            # Continue with rest of cleanup even if disk cleanup fails

        # Also clean up old background tasks
        if len(batch_ids_to_remove) > 10:
            # If we're cleaning up a lot of results, also clean up background tasks
            tasks_to_remove = []
            for task_id, task in _background_tasks.items():
                if task_id == _embedding_task_key:
                    continue  # Skip the embedding task

                timestamp = task.get("timestamp", 0)
                if current_time - timestamp > retention_period:
                    tasks_to_remove.append(task_id)

            if tasks_to_remove:
                async with _tasks_lock:
                    for task_id in tasks_to_remove:
                        if task_id in _background_tasks:
                            del _background_tasks[task_id]

                logger.info(f"Cleaned up {len(tasks_to_remove)} old background tasks")

    except Exception as e:
        logger.error(f"Error cleaning up old results: {str(e)}")
        logger.error(traceback.format_exc())

async def _update_batch_history(updates: Dict[str, Dict[str, Any]]) -> None:
    """Update the batch history file with task status information.

    This maintains a record of all batch tasks even after they're removed from memory.

    Args:
        updates: Dictionary mapping batch IDs to their status information
    """
    try:
        history_path = Path(Config.get_paths()['logs']) / "batch_history.json"
        history_lock_path = history_path.with_suffix('.lock')
        history = {}

        # Create directory if it doesn't exist
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Use file lock to prevent concurrent writes
        async with asyncio.timeout(10):  # 10 second timeout
            with FileLock(str(history_lock_path), timeout=5):
                # Load existing history if available
                if history_path.exists():
                    try:
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading batch history: {e}")
                        # If file is corrupted, start fresh
                        history = {}

                # Update history with new information
                history.update(updates)

                # Limit history size to prevent unlimited growth
                if len(history) > 1000:
                    # Keep only the most recent 1000 entries
                    sorted_entries = sorted(
                        history.items(), 
                        key=lambda x: (
                            x[1].get('removed_at', ''), 
                            x[1].get('completed_at', ''), 
                            x[1].get('created_at', '')
                        ),
                        reverse=True
                    )
                    history = dict(sorted_entries[:1000])

                # Write updated history to file
                temp_path = history_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(history, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Move temp file to final location for atomic write
                shutil.move(temp_path, history_path)

                logger.debug(f"Updated batch history with {len(updates)} entries")
    except Exception as e:
        logger.error(f"Error updating batch history: {e}")

def _generate_task_id(prefix: str = "task") -> str:
    """Generate a unique task ID with the given prefix.

    Args:
        prefix: Prefix for the task ID (e.g., 'query', 'batch')

    Returns:
        A unique task ID string
    """
    timestamp = int(time.time())
    random_suffix = secrets.token_hex(4)
    return f"{prefix}_{timestamp}_{random_suffix}"

def _is_valid_task_id(task_id: str) -> bool:
    """Validate a task ID format.

    Args:
        task_id: The task ID to validate

    Returns:
        True if the task ID is valid, False otherwise
    """
    # Allow alphanumeric characters, underscores, and hyphens
    # Length must be between 4 and 64 characters
    pattern = re.compile(r'^[a-zA-Z0-9_-]{4,64}$')
    return bool(pattern.match(task_id))

def _check_task_id_collision(task_id: str) -> bool:
    """Check if a task ID already exists in the system.

    Args:
        task_id: The task ID to check

    Returns:
        True if the task ID exists, False otherwise
    """
    # Check in active background tasks
    if task_id in _background_tasks:
        return True

    # Check in stored batch results
    if task_id in _batch_results:
        return True

    # Check in query batch
    for item in _query_batch:
        if item.get("id") == task_id:
            return True

    return False

@router.post("/api/v1/data/prepare")
async def prepare_data(request: Request):
    """Prepare data for inference."""
    try:
        config = ChanScopeConfig.from_env()

        # Initialize storage with environment detection
        from config.storage import StorageFactory
        storage = StorageFactory.create(config)

        # TODO: Implement this
    except Exception as e:
        handle_error(e)

@router.post("/api/v1/data/stratify")
async def stratify_data(request: Request):
    """Create stratified sample."""
    try:
        # Initialize storage with environment detection
        from config.storage import StorageFactory
        storage = StorageFactory.create(data_ops.config)

        # TODO: Implement this
    except Exception as e:
        handle_error(e)

# Define request models
class NLQueryRequest(BaseModel):
    """Natural language query request model."""
    query: str
    limit: Optional[int] = 100  # Default limit of 100 records
    provider: Optional[str] = None  # This parameter is ignored; static providers are always used
    format_for_llm: Optional[bool] = True  # Whether to format the response for LLM consumption

    class Config:
        """Pydantic model configuration."""
        schema_extra = {
            "example": {
                "query": "Find 5 rows from the last 12 hours containing tarrif",
                "limit": 5,
                "format_for_llm": True
            }
        }

class NLQueryResponse(BaseModel):
    """Natural language query response model."""
    status: str = "success"
    query: str
    description: Dict[str, Any]
    sql: str
    record_count: int
    data: List[Dict[str, Any]]
    execution_time_ms: float
    metadata: Dict[str, Any] = {}

    class Config:
        """Pydantic model configuration."""
        # Allow for arbitrary types like numpy to be converted to JSON-compatible formats
        json_encoders = {
            # Convert numpy types to Python native types
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            # Ensure dates are formatted consistently
            datetime: lambda x: x.isoformat(),
            date: lambda x: x.isoformat(),
        }
        # Populate by field name to ensure consistency
        populate_by_name = True
        # Schema configuration for better OpenAPI documentation
        schema_extra = {
            "example": {
                "status": "success",
                "query": "Find 5 rows from the last 12 hours containing tarrif",
                "description": {
                    "original_query": "Find 5 rows from the last 12 hours containing tarrif",
                    "query_time": "2025-04-14T18:25:17.987595+00:00",
                    "filters": [
                        "Time: Last 12 hours", 
                        "Limit: 5 rows",
                        "Content: Contains 'tarrif'"
                    ],
                    "time_filter": "Last 12 hours",
                    "limit": 5,
                    "content_filter": "tarrif"
                },
                "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s AND content ILIKE %s ORDER BY posted_date_time DESC LIMIT 5",
                "record_count": 5,
                "data": [
                    {
                        "id": 385250,
                        "content": "Discussion about recent tarrif changes affecting trade policies and economic outlook.",
                        "posted_date_time": "2025-04-14T13:31:41+00:00",
                        "thread_id": "17647448"
                    },
                    {
                        "id": 385251,
                        "content": "The new tarrif implementation has sparked debate among economists.",
                        "posted_date_time": "2025-04-14T13:30:46+00:00",
                        "thread_id": "17647447"
                    }
                ],
                "execution_time_ms": 189.72,
                "metadata": {
                    "processing_time_ms": 189.72,
                    "sql_generation_method": "llm",
                    "timestamp": "2025-04-14T18:25:18.177317+00:00",
                    "providers_used": {
                        "enhancer": "openai",
                        "generator": "venice"
                    }
                }
            }
        }

@router.post("/nl_query", response_model=NLQueryResponse)
async def natural_language_query(
    request: NLQueryRequest,
    agent: KnowledgeAgent = Depends(get_agent),
    save_result: bool = Query(True, description="Whether to save query results to disk")
) -> Dict[str, Any]:
    """
    Process a natural language query against the database using LLM-generated SQL.

    This endpoint converts a natural language query string to SQL using a three-stage LLM process
    and executes it against the complete_data table in the PostgreSQL database.

    The SQL generation uses fixed providers:
    - OpenAI for query enhancement
    - Venice for SQL generation and validation

    Args:
        request: The NLQueryRequest containing the natural language query
        agent: KnowledgeAgent instance
        save_result: Whether to save query results to disk

    Returns:
        JSON response with query results and metadata

    Example queries:
        - "Give me threads from the last hour"
        - "Show posts from yesterday containing crypto"
        - "Find messages from the last 3 days by author john"
        - "Get threads from board tech about AI from this week"
        - "Find 5 rows from the last 12 hours containing tarrif"
        - "Find 8 rows from the last 24 hours that contains bitcoin"
        - "Show me 10 random posts mentioning ethereum"
    """
    start_time = time.time()

    # Check if we're in a Replit environment
    from config.env_loader import detect_environment, is_replit_environment
    env_type = detect_environment()

    if not is_replit_environment():
        # Return a helpful error message for Docker/local environments
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ENVIRONMENT_RESTRICTION",
                "message": "Natural language queries are only available in the Replit environment where PostgreSQL is properly configured.",
                "environment": env_type,
                "possible_solution": "This feature requires a PostgreSQL database with the complete_data table. Please check the documentation for setup instructions."
            }
        )

    try:
        logger.info(f"Processing natural language query: {request.query}")

        # Create SQL generator
        sql_generator = LLMSQLGenerator(agent)

        # Get the query description for user feedback
        description = sql_generator.get_query_description(request.query)

        # Convert NL to SQL - Note that provider parameter is ignored, static providers are used
        sql_query, params = await sql_generator.generate_sql(
            nl_query=request.query,
            provider=None,  # This is ignored, static providers are used
            use_hybrid_approach=True  # Use template matching for common patterns
        )

        # Apply limit if not already in the query
        if request.limit and "LIMIT" not in sql_query.upper():
            sql_query += f" LIMIT %s"
            params.append(request.limit)

        # Initialize the database connection
        from config.replit import PostgresDB
        db = PostgresDB()

        # Execute the query
        with db.get_connection() as conn:
            # Use pandas to read from database with parameterized query
            try:
                logger.info(f"Executing SQL query: {sql_query}")
                logger.debug(f"With parameters: {params}")
                df = pd.read_sql(sql_query, conn, params=params)
                logger.info(f"Query returned {len(df)} rows")
            except Exception as e:
                logger.error(f"Error executing SQL query: {e}")
                logger.error(traceback.format_exc())
                raise ProcessingError(
                    message=f"Error executing SQL query: {str(e)}",
                    operation="sql_execution"
                )

        # Convert to dictionaries for JSON response
        # Handle datetime columns by converting to ISO format strings
        records = []
        for _, row in df.iterrows():
            record = {}
            for column, value in row.items():
                # Skip less valuable fields
                if column in ['author', 'channel_name', 'inserted_at']:
                    continue

                if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                    record[column] = value.isoformat()
                else:
                    record[column] = value

                # Trim long content to a reasonable length for LLM consumption
                if column == 'content' and isinstance(value, str) and len(value) > 500:
                    record[column] = value[:500] + "..."

            records.append(record)

        # Calculate execution time
        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        # Build metadata
        metadata = {
            "processing_time_ms": execution_time_ms,
            "sql_generation_method": "llm",
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "providers_used": {
                "enhancer": sql_generator.PROVIDER_ENHANCER.value,
                "generator": sql_generator.PROVIDER_GENERATOR.value
            }
        }

        # Construct response
        response = {
            "status": "success",
            "query": request.query,
            "description": description,
            "sql": sql_query,
            "record_count": len(records),
            "data": records,
            "execution_time_ms": execution_time_ms,
            "metadata": metadata
        }

        # Save result to disk if requested
        if save_result:
            try:
                # Generate a unique task ID for this query
                task_id = _generate_task_id(prefix="nlquery")
                
                # Save response to disk
                from knowledge_agents.utils import save_query_output
                base_path = Path(Config.get_paths().get('generated_data', 'data/generated_data'))
                
                # Structure the response for saving
                response_for_saving = {
                    "task_id": task_id,
                    "query": request.query,
                    "sql": sql_query,
                    "description": description,
                    "data": records,
                    "metadata": metadata.copy()
                }
                
                # Save to disk
                json_path, _ = await save_query_output(
                    response=response_for_saving,
                    base_path=base_path,
                    task_id=task_id,
                    query=request.query,
                    logger=logger,
                    include_embeddings=False  # NL queries don't use embeddings
                )
                
                # Add saved file paths to response metadata
                if json_path:
                    if "saved_files" not in response["metadata"]:
                        response["metadata"]["saved_files"] = {}
                    response["metadata"]["saved_files"]["json"] = str(json_path)
                    response["metadata"]["task_id"] = task_id
                    
                    # Add storage type to metadata
                    from config.env_loader import is_replit_environment
                    response["metadata"]["storage_type"] = "object_storage" if is_replit_environment() else "filesystem"
                    
                    logger.info(f"Saved NL query results to {json_path}")
                
            except Exception as save_error:
                logger.warning(f"Could not save NL query results to disk: {save_error}")
                # Continue even if saving fails - don't affect the overall operation

        # Format the response for better LLM readability if requested
        if request.format_for_llm:
            response = format_response_for_llm(response)

        # Log endpoint call
        log_endpoint_call(
            logger=logger, 
            endpoint="/nl_query",
            method="POST", 
            duration_ms=execution_time_ms,
            params={"query": request.query, "limit": request.limit, "record_count": len(records)}
        )

        return response

    except NLQueryParsingError as e:
        # Handle parsing errors
        error = ValidationError(
            message=f"Could not parse natural language query: {str(e)}",
            field="query",
            value=request.query
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )
    except ProcessingError as e:
        # Forward processing errors
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
        # Handle unexpected errors
        error = APIError(
            message=f"Error processing natural language query: {str(e)}",
            status_code=500,
            error_code="NL_QUERY_ERROR",
            details={"query": request.query, "error": str(e)}
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

def format_response_for_llm(response: Dict[str, Any]) -> Dict[str, Any]:
    """Format response data to be more friendly for LLM consumption."""
    # Create a copy to avoid modifying the original
    formatted = copy.deepcopy(response)

    # Ensure consistent spacing and formatting in the response
    if "data" in formatted and isinstance(formatted["data"], list):
        # Ensure each record has the same keys in the same order
        if formatted["data"]:
            # Get all possible keys from all records
            all_keys = set()
            for record in formatted["data"]:
                all_keys.update(record.keys())

            # Sort keys for consistent order (id first, then content, then others alphabetically)
            sorted_keys = sorted(all_keys)
            if "id" in sorted_keys:
                sorted_keys.remove("id")
                sorted_keys = ["id"] + sorted_keys
            if "content" in sorted_keys:
                sorted_keys.remove("content")
                sorted_keys.insert(1, "content")
            if "posted_date_time" in sorted_keys:
                sorted_keys.remove("posted_date_time")
                sorted_keys.insert(2, "posted_date_time")

            # Reorder all records to have the same keys in the same order
            formatted_data = []
            for record in formatted["data"]:
                formatted_record = {}
                for key in sorted_keys:
                    if key in record:
                        formatted_record[key] = record[key]
                formatted_data.append(formatted_record)

            formatted["data"] = formatted_data

    return formatted

@router.post("/api/v1/nl_query", response_model=NLQueryResponse)
async def nl_query_api_v1(
    request: NLQueryRequest,
    agent: KnowledgeAgent = Depends(get_agent),
    save_result: bool = Query(True, description="Whether to save query results to disk")
) -> Dict[str, Any]:
    """API v1 endpoint for natural language queries."""
    return await natural_language_query(request, agent, save_result)

@router.post("/admin/cleanup")
async def trigger_cleanup(
    force: bool = Query(False, description="Force cleanup of all files regardless of age")
) -> Dict[str, Any]:
    """Admin endpoint to manually trigger cleanup of memory and disk storage."""
    start_time = time.time()
    memory_items_removed = 0
    disk_files_removed = 0
    
    try:
        # Memory cleanup
        current_time = time.time()
        retention_period = 600  # 10 minutes for memory items
        
        # Count items before cleanup
        memory_items_before = len(_batch_results)
        
        # Get list of expired items
        batch_ids_to_remove = []
        for batch_id, result in _batch_results.items():
            timestamp = result.get("metadata", {}).get("timestamp", 0)
            if force or (current_time - timestamp > retention_period):
                batch_ids_to_remove.append(batch_id)
        
        # Remove items
        for batch_id in batch_ids_to_remove:
            if batch_id in _batch_results:
                del _batch_results[batch_id]
        
        memory_items_removed = len(batch_ids_to_remove)
        
        # Disk cleanup
        disk_retention_period = 86400  # 1 day
        base_path = Path(Config.get_paths().get('generated_data', 'data/generated_data'))
        
        if base_path.exists():
            # First count total files
            total_files = sum(1 for _ in base_path.glob("**/*.*"))
            
            # Process files
            for json_file in base_path.glob("**/*.json"):
                try:
                    # Get file modification time
                    file_mtime = json_file.stat().st_mtime
                    file_age = current_time - file_mtime
                    
                    # Delete files based on age or force parameter
                    if force or file_age > disk_retention_period:
                        # Check for corresponding embedding files
                        embedding_json = base_path / "embeddings" / f"{json_file.stem}_embeddings.json"
                        embedding_npz = base_path / "embeddings" / f"{json_file.stem}_embeddings.npz"
                        
                        # Delete JSON file
                        json_file.unlink()
                        disk_files_removed += 1
                        
                        # Delete embedding files if they exist
                        if embedding_json.exists():
                            embedding_json.unlink()
                            disk_files_removed += 1
                        if embedding_npz.exists():
                            embedding_npz.unlink()
                            disk_files_removed += 1
                except Exception as file_error:
                    logger.warning(f"Error cleaning up file {json_file}: {str(file_error)}")
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Manual cleanup completed in {duration_ms}ms: {memory_items_removed} memory items, {disk_files_removed} disk files")
        
        return {
            "status": "success",
            "memory_cleanup": {
                "items_before": memory_items_before,
                "items_removed": memory_items_removed,
                "items_remaining": len(_batch_results)
            },
            "disk_cleanup": {
                "files_removed": disk_files_removed,
                "retention_period_hours": int(disk_retention_period / 3600),
                "force_applied": force
            },
            "duration_ms": duration_ms
        }
    
    except Exception as e:
        logger.error(f"Error during manual cleanup: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "memory_items_removed": memory_items_removed,
            "disk_files_removed": disk_files_removed,
            "duration_ms": round((time.time() - start_time) * 1000, 2)
        }

