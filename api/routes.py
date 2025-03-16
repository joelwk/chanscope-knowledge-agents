"""API routes and endpoint handlers."""
import time
import traceback
import os
import yaml
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

import json
import shutil
from filelock import FileLock
import pytz
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import secrets
import re

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
    build_model_config)

from config.logging_config import get_logger
from api.models import HealthResponse, StratificationResponse, log_endpoint_call
from api.cache import CACHE_HITS, CACHE_MISSES, CACHE_ERRORS, cache

from config.env_loader import is_replit_environment, get_replit_paths
from knowledge_agents.utils import save_query_output

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
        "debug_request": "/debug/request"
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

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    start_time = time.time()
    try:
        # Get environment information
        env_info = {
            "docker_env": Config.get_api_settings()['docker_env'],
            "service_type": os.getenv("SERVICE_TYPE", "unknown"),
            "api_version": "1.0.0",
            "is_replit": is_replit_environment(),
            "replit_env": os.getenv("REPLIT_ENV", ""),
            "repl_id": os.getenv("REPL_ID", "")
        }

        # Check data directories
        paths = get_replit_paths() if is_replit_environment() else {}
        data_status = {
            "root_data_exists": os.path.exists(paths.get("root_data_path", "/app/data")),
            "stratified_data_exists": os.path.exists(paths.get("stratified_path", "/app/data/stratified")),
            "logs_exists": os.path.exists(paths.get("logs_path", "/app/logs"))
        }

        response = HealthResponse(
            status="healthy",
            message="Service is running",
            timestamp=datetime.now(pytz.UTC),
            environment=env_info,
            data_status=data_status
        )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_endpoint_call(
            logger=logger,
            endpoint="/health",
            method="GET",
            duration_ms=duration_ms,
            params={"environment": env_info}
        )

        return response
    except Exception as e:
        error = APIError(
            message="Health check failed",
            status_code=500,
            error_code="HEALTH_CHECK_ERROR",
            details={"error": str(e)}
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

@router.get("/healthz", include_in_schema=False)
async def healthz():
    """Simple health check endpoint for Replit's health check system."""
    return {
        "status": "ok", 
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "environment": os.getenv("REPLIT_ENV", "development")
    }

@router.get("/health_replit", response_model=dict)
async def health_check_replit():
    """Extended health check with Replit-specific info."""
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

    # Get port configuration
    port = os.getenv('PORT', '80')
    api_port = os.getenv('API_PORT', port)
    host = os.getenv('HOST', '0.0.0.0')
    
    # Check if running in Replit environment
    is_replit = replit_env in ['true', 'replit', 'production']
    
    # Get additional environment information
    environment_vars = {
        "is_replit": is_replit,
        "replit_env": replit_env,
        "repl_id": replit_id,
        "repl_slug": replit_slug,
        "repl_owner": replit_owner,
        "replit_dev_domain": replit_dev_domain,
        "python_path": os.getenv('PYTHONPATH', ''),
        "fastapi_env": os.getenv('FASTAPI_ENV', ''),
        "fastapi_debug": os.getenv('FASTAPI_DEBUG', '')
    }
    
    # Get service configuration
    service_config = {
        "url": service_url,
        "port": port,
        "api_port": api_port,
        "host": host,
        "api_url": f"{service_url}/api" if service_url else None,
        "api_v1_url": f"{service_url}/api/v1" if service_url else None,
        "api_base_path": os.getenv('API_BASE_PATH', '/api/v1'),
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
        
        # Get configuration
        config = build_model_config()
        
        if request.use_background:
            # Add task to background processing
            background_tasks.add_task(
                _process_single_query,
                task_id=task_id,
                query=request.query,
                agent=agent,
                config=config,
                use_batching=True,
                data_ops=data_ops,
                force_refresh=request.force_refresh,
                skip_embeddings=request.skip_embeddings
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
                # First ensure data is ready if needed
                if request.force_refresh or not await data_ops.is_data_ready(skip_embeddings=request.skip_embeddings):
                    await data_ops.ensure_data_ready(
                        force_refresh=request.force_refresh,
                        skip_embeddings=request.skip_embeddings
                    )
                
                # Load the stratified data
                library_df = await data_ops.load_stratified_data()
                
                # Now process the query with the loaded data
                result = await process_query(
                    query=request.query,
                    agent=agent,
                    library_df=library_df,
                    config=config
                )
                
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
                    json_path, embeddings_path = save_query_output(
                        response=response,
                        base_path=base_path,
                        logger=logger,
                        query=request.query
                    )
                    
                    # Add file paths to response metadata
                    if "metadata" not in response:
                        response["metadata"] = {}
                    response["metadata"]["saved_files"] = {
                        "json": str(json_path)
                    }
                    if embeddings_path:
                        response["metadata"]["saved_files"]["embeddings"] = str(embeddings_path)
                        
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
    use_batching: bool,
    data_ops: DataOperations,
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> None:
    """Process a single query and store the results."""
    try:
        # Update task status to processing
        async with _tasks_lock:
            _background_tasks[task_id] = {
                "status": "processing",
                "timestamp": time.time(),
                "query": query,
                "progress": 10
            }
        
        # Process the query
        # First ensure data is ready if needed
        if force_refresh or not await data_ops.is_data_ready(skip_embeddings=skip_embeddings):
            await data_ops.ensure_data_ready(
                force_refresh=force_refresh,
                skip_embeddings=skip_embeddings
            )
        
        # Load the stratified data
        library_df = await data_ops.load_stratified_data()
        
        # Now process the query with the loaded data
        result = await process_query(
            query=query,
            agent=agent,
            library_df=library_df,
            config=config
        )
        
        # Create complete response
        response = {
            "status": "completed",
            "task_id": task_id,
            "chunks": result.get("chunks", []),
            "summary": result.get("summary", ""),
            "metadata": result.get("metadata", {})
        }
        
        # Save complete response
        try:
            base_path = Path(Config.get_paths()["generated_data"])
            json_path, embeddings_path = save_query_output(
                response=response,
                base_path=base_path,
                logger=logger,
                query=query
            )
            
            # Add file paths to response metadata
            if "metadata" not in response:
                response["metadata"] = {}
            response["metadata"]["saved_files"] = {
                "json": str(json_path)
            }
            if embeddings_path:
                response["metadata"]["saved_files"]["embeddings"] = str(embeddings_path)
                
        except Exception as e:
            logger.error(f"Error saving query output: {e}")
            # Continue processing even if saving fails
        
        # Store the result
        success = await _store_batch_result(
            batch_id=task_id,
            result=response,  # Pass the complete response
            config=config
        )
        
        if success:
            processing_time = result.get("metadata", {}).get("processing_time_ms", 0)
            logger.info(f"Query {task_id} processed successfully in {processing_time/1000:.2f}s")
            
            # Update task status to completed
            async with _tasks_lock:
                if task_id in _background_tasks:
                    _background_tasks[task_id]["status"] = "completed"
                    _background_tasks[task_id]["progress"] = 100
        else:
            logger.error(f"Failed to store results for query {task_id}")
            
            # Update task status to failed
            async with _tasks_lock:
                if task_id in _background_tasks:
                    _background_tasks[task_id]["status"] = "failed"
                    _background_tasks[task_id]["error"] = "Failed to store results"
            
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
            }
        )
        
        # Log processing time
        duration_ms = round((time.time() - start_time) * 1000, 2)
        avg_time_per_query = round(duration_ms / len(request.queries), 2)
        logger.info(f"Batch processed {len(request.queries)} queries in {duration_ms}ms (avg: {avg_time_per_query}ms/query)")
        
        # Return results with metadata
        return {
            "batch_id": batch_id,
            "results": results,
            "metadata": {
                "total_time_ms": duration_ms,
                "avg_time_per_query_ms": avg_time_per_query,
                "queries_processed": len(results),
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
    task_id: Optional[str] = None
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
    """
    try:
        # Calculate time range
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(hours=6)
        filter_date = start_time.strftime('%Y-%m-%d %H:%M:%S+00:00')

        # Get base settings and paths
        base_settings = get_base_settings()
        paths = base_settings.get('paths', {})
        stored_queries_path = Path(paths.get('config', 'config')) / 'stored_queries.yaml'

        # Load stored query
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
            filter_date=filter_date,
            force_refresh=False,
            skip_embeddings=True,
            skip_batching=False,
            select_board=select_board,
            task_id=task_id
        )

        # Log request
        log_params = {
            "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "filter_date": filter_date,
            "select_board": select_board
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
        if isinstance(result, dict) and "status" in result and result["status"] == "queued":
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
    """Check health of the in-memory cache."""
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
    """Prepare data for inference with enhanced reliability and performance.
    
    This function implements the Chanscope approach for data preparation:
    
    For force_refresh=True:
    - Verify that complete_data.csv exists and is updated with latest S3 data
    - Create a new stratified sample and generate new embeddings
    - Search embeddings for related chunks
    
    For force_refresh=False:
    - Check if complete_data.csv exists (but don't verify updates)
    - Use existing stratified data and embeddings
    - If data doesn't exist, proceed as if force_refresh=True
    
    Args:
        config: Model configuration with processing settings
        data_ops: Data operations for data preparation
        force_refresh: Whether to force data refresh
        skip_embeddings: Whether to skip embedding generation (for testing)
        
    Returns:
        DataFrame containing prepared stratified data
        
    Raises:
        HTTPException: If data preparation fails
        ProcessingError: If specific processing steps fail
    """
    start_time = time.time()
    try:
        logger.info(f"Preparing data (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})")
        
        # Step 1: Check data freshness to determine if update is needed
        data_freshness_check = await data_ops.check_data_freshness()
        
        # Get paths for validation
        complete_data_path = data_ops.config.root_data_path / "complete_data.csv"
        stratified_path = data_ops.config.stratified_data_path / "stratified_sample.csv"
        embeddings_path = data_ops.config.stratified_data_path / "embeddings.npz"
        
        # Case A: force_refresh=True - Follow Chanscope approach for forced refresh
        if force_refresh:
            logger.info("Force refresh requested, implementing Chanscope forced refresh approach")
            
            # Check if complete_data.csv exists and is up-to-date with S3
            # Only refresh if it doesn't exist or isn't current
            if not complete_data_path.exists():
                logger.info("Complete data file doesn't exist, fetching from source")
                needs_data_update = True
            else:
                # Check if data is current with S3
                is_current = await data_ops._is_data_up_to_date_with_s3()
                needs_data_update = not is_current
                logger.info(f"Complete data exists, is current with S3: {is_current}")
            
            # If data needs update, use ensure_data_ready with force_refresh=True
            # to properly fetch and process according to the Chanscope approach
            if needs_data_update:
                logger.info("Data needs update, performing complete refresh")
                await data_ops.ensure_data_ready(force_refresh=True, skip_embeddings=skip_embeddings)
            else:
                logger.info("Complete data is current, performing stratification and embedding refresh only")
                # Even if data is current, with force_refresh=True we still refresh stratification and embeddings
                # Create new stratified sample
                await data_ops._create_stratified_dataset()
                
                # Generate new embeddings unless explicitly skipped
                if not skip_embeddings:
                    await data_ops.generate_embeddings(force_refresh=True)
        
        # Case B: force_refresh=False - Follow Chanscope approach for regular processing
        else:
            logger.info("Using existing data (force_refresh=false), checking data existence")
            
            # Check if complete_data.csv exists
            if not complete_data_path.exists():
                logger.info("Complete data file doesn't exist, proceeding as if force_refresh=True")
                # If it doesn't exist, proceed as if force_refresh=True 
                await data_ops.ensure_data_ready(force_refresh=True, skip_embeddings=skip_embeddings)
            else:
                logger.info("Complete data exists, checking stratified data and embeddings")
                
                # Check if stratified data exists
                if not stratified_path.exists():
                    logger.info("Stratified data doesn't exist, creating it")
                    await data_ops._create_stratified_dataset()
                
                # Check if embeddings exist and generate if needed (unless skipped)
                if not skip_embeddings and not embeddings_path.exists():
                    logger.info("Embeddings don't exist, generating them")
                    await data_ops.generate_embeddings(force_refresh=False)
        
        # Load stratified data for use in inference
        stratified_data = await data_ops._load_stratified_data()
        
        # Check data validity
        if stratified_data is None or stratified_data.empty:
            raise ProcessingError(
                message="Stratified data is empty or not available",
                operation="load_stratified_data"
            )
        
        data_rows = len(stratified_data)
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Data preparation complete: {data_rows} rows available (took {processing_time}ms)")
        
        # Return data for use in inference
        return stratified_data
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {str(e)}")
        raise ProcessingError(
            message=f"Required data file not found: {str(e)}",
            operation="data_preparation",
            resource=str(e.filename) if hasattr(e, 'filename') else None,
            original_error=e
        )
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file: {str(e)}")
        raise ProcessingError(
            message="Data file is empty",
            operation="data_preparation",
            original_error=e
        )
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}", exc_info=True)
        raise ProcessingError(
            message=f"Data preparation failed: {str(e)}",
            operation="data_preparation",
            original_error=e
        )


async def _store_batch_result(batch_id: str, result: Union[tuple, Dict[str, Any]], config: Union[Dict[str, Any], ModelConfig]) -> bool:
    """Store batch processing results for later retrieval.
    
    Args:
        batch_id: Unique identifier for the task (format: prefix_timestamp_randomhex)
        result: Either a tuple of (chunks, summary) or a dict with chunks, summary, and metadata
        config: Either a ModelConfig object or a dictionary with configuration
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
            metadata = result.get("metadata", {})

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
            logger.warning("Redis client not available, skipping result storage")
        
        # Update batch history
        await _update_batch_history({batch_id: {'timestamp': time.time(), 'query': config.query if hasattr(config, 'query') else ''}})
        
        logger.info(f"Stored result for task {batch_id} (expires in {batch_result_ttl}s)")
        
        # Also store results in background tasks for redundancy
        if batch_id in _background_tasks:
            _background_tasks[batch_id]['results'] = result
        
        # Store in memory cache
        _batch_results[batch_id] = result_obj
        
        return True
        
    except Exception as e:
        logger.error(f"Error storing result: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
async def _cleanup_old_results():
    """Clean up old results to prevent memory leaks."""
    try:
        retention_period = 600  # 10 minutes
        current_time = time.time()
        batch_ids_to_remove = []
        batch_history_updates = {}
        
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
        
        # Remove expired results
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
            
            logger.info(f"Cleaned up {len(batch_ids_to_remove)} old results (retention: {retention_period}s)")
            if memory_freed > 0:
                logger.info(f"Estimated memory freed: {memory_freed / 1024:.2f} KB")
        
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

