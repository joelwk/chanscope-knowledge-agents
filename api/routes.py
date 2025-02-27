"""API routes and endpoint handlers."""
import time
import traceback
import os
import gc
import yaml
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Set, Awaitable, Callable
from collections import deque
import hashlib
import json
import shutil
from filelock import FileLock
import numpy as np

import pytz
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from knowledge_agents.model_ops import (
    ModelProvider, 
    KnowledgeAgent, 
    ModelProviderError,
    ModelConfigurationError,
    ModelOperationError,
    ModelOperation,
    ModelConfig)
 
from knowledge_agents.embedding_ops import get_agent
from knowledge_agents.data_processing.cloud_handler import S3Handler
from knowledge_agents.data_ops import DataConfig, DataOperations
from knowledge_agents.inference_ops import process_multiple_queries
from knowledge_agents.run import run_inference

from config.base_settings import get_base_settings
from config.settings import Config
from config.config_utils import (
    QueryRequest,
    BatchQueryRequest,
    build_unified_config)

from config.logging_config import get_logger
from api.models import HealthResponse, StratificationResponse, log_endpoint_call
from api.cache import CACHE_HITS, CACHE_MISSES, CACHE_ERRORS, cache

from api.errors import ProcessingError

logger = get_logger(__name__)
router = APIRouter()

# Global dictionary to track background tasks with lock
_background_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = asyncio.Lock()
_embedding_task_key = 'embedding_generation'

# Batch processing configuration
_query_batch: deque = deque()
_query_batch_lock = asyncio.Lock()
_batch_results: Dict[str, Any] = {}
_BATCH_SIZE = Config.get_model_settings()['embedding_batch_size']  # Batch size from configuration
_BATCH_WAIT_TIME = 2  # Seconds to wait for batch accumulation
_last_batch_processing_time = 0  # Track processing time for ETA estimates

# Enhanced error handling classes
class APIError(Exception):
    """Base exception for API errors with structured logging."""
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured dictionary for logging and response."""
        error_dict = {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        if self.original_error:
            error_dict["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            }
        return error_dict

    def log_error(self, logger: logging.Logger):
        """Log error with consistent structure."""
        error_dict = self.to_dict()
        logger.error(
            f"{self.error_code}: {self.message}",
            extra={
                "error_details": error_dict,
                "status_code": self.status_code,
                "traceback": traceback.format_exc() if self.original_error else None
            }
        )

class ValidationError(APIError):
    """Exception for request validation errors."""
    def __init__(
        self, 
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        details = {"field": field, "value": value} if field else {}
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )

class ConfigurationError(APIError):
    """Exception for configuration errors."""
    def __init__(
        self, 
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None
    ):
        details = {
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None
        } if config_key else {}
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details
        )

# Wrapper function to replace direct calls to _run_inference_async
async def _run_inference_async(
    query: str,
    data_config: ModelConfig,
    agent: KnowledgeAgent,
    force_refresh: bool = False,
    skip_embeddings: bool = False
):
    """
    Wrapper function to call run_inference from knowledge_agents.run
    
    Args:
        query: The query to process
        data_config: Configuration for data processing
        agent: KnowledgeAgent instance to use for processing
        force_refresh: Whether to force refresh the data
        skip_embeddings: Whether to skip embedding generation
        
    Returns:
        Tuple[List[Dict[str, Any]], str]: Chunks and summary
    """
    # Call run_inference which will internally call the original _run_inference_async
    result = run_inference(
        query=query,
        config=data_config,
        agent=agent,
        force_refresh=force_refresh,
        skip_embeddings=skip_embeddings
    )
    
    # If run_inference returned a coroutine (Future), await it
    if asyncio.isfuture(result) or asyncio.iscoroutine(result):
        return await result
    
    # Otherwise, return the result directly
    return result

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
        "process_query": "/process_query",
        "batch_process": "/batch_process",
        "stratify": "/stratify",
        "process_recent_query": "/process_recent_query",
        "health_replit": "/health_replit",
        "connections": "/health/connections",
        "s3_health": "/health/s3",
        "provider_health": "/health/provider/{provider}",
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
            "api_version": "1.0.0"
        }

        response = HealthResponse(
            status="healthy",
            message="Service is running",
            timestamp=datetime.now(pytz.UTC),
            environment=env_info
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
    
    return {
        "status": "healthy",
        "message": "Replit Service is running",
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "environment": environment_vars,
        "service": service_config,
        "system": system_info,
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
            params={"provider": provider, "response": response}
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

@router.post("/process_query")
async def process_query(
    request: QueryRequest,
    agent: KnowledgeAgent = Depends(get_agent),
    data_ops: DataOperations = Depends(get_data_ops),
    background_tasks: BackgroundTasks = None
):
    """Process a query with optimized batching and parallel processing.
    
    This function serves as the main entry point for query processing, with two paths:
    1. Direct processing (skip_batching=True): Processes the query immediately
    2. Batch processing: Adds query to a batch queue for efficient processing
    
    Args:
        request: The query request with parameters
        agent: Knowledge agent for processing
        data_ops: Data operations for data preparation
        background_tasks: Optional background tasks for async processing
        
    Returns:
        Either the query results or a batch status object
    """
    start_time = time.time()
    try:
        # Generate a unique request ID for tracking and caching
        request_id = hashlib.md5(
            f"{request.query}:{request.filter_date or ''}:{request.embedding_provider or ''}:{request.chunk_provider or ''}:{request.select_board or ''}"
            .encode()
        ).hexdigest()
        cache_key = f"query_result:{request_id}"
        
        # Try to get from cache first if not forcing refresh
        if not request.force_refresh:
            try:
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for query: {request_id[:8]}...")
                    return cached_result
            except Exception as cache_error:
                # Log cache error but continue processing
                logger.warning(f"Cache retrieval error: {str(cache_error)}")
        
        # Process immediately if batching is skipped
        if request.skip_batching:
            logger.info(f"Processing query immediately (skip_batching=True): {request_id[:8]}...")
            
            # Build unified configuration
            config = build_unified_config(request)
            
            # Prepare data if needed - ensures stratified data is ready for embeddings and search
            await _prepare_data_if_needed(
                config=config,
                data_ops=data_ops,
                force_refresh=request.force_refresh,
                skip_embeddings=request.skip_embeddings
            )
            
            # Run inference using the prepared data
            chunks, summary = await _run_inference_async(
                request.query,
                config,
                agent,
                request.force_refresh,
                request.skip_embeddings
            )
            
            # Construct and cache the result
            result = {"chunks": chunks, "summary": summary}
            try:
                await cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour
            except Exception as cache_error:
                # Log cache error but continue since we have the result
                logger.warning(f"Cache storage error: {str(cache_error)}")
            
            # Log query processing time
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Query processed in {duration_ms}ms: {request_id[:8]}...")
            
            return result
        else:
            # Add to batch queue for efficient processing
            if background_tasks is None:
                # If no background_tasks provided, return error
                raise ValidationError(
                    message="Background tasks unavailable for batch processing",
                    field="background_tasks",
                    value=None
                )
            
            # Add to query batch and get batch ID
            batch_id = await _add_to_query_batch(request)
            logger.info(f"Added query to batch: {batch_id}")
            
            # Schedule batch processing if needed
            if _should_process_batch():
                logger.info(f"Batch processing triggered by batch size threshold")
                background_tasks.add_task(_process_query_batch, agent, data_ops)
            else:
                # Schedule processing after wait time
                async def delayed_process():
                    await asyncio.sleep(_BATCH_WAIT_TIME)
                    if _should_process_batch():
                        await _process_query_batch(agent, data_ops)
                
                logger.info(f"Scheduled delayed batch processing in {_BATCH_WAIT_TIME}s")
                background_tasks.add_task(delayed_process)
                
            # Return batch status information
            queue_position = len(_query_batch) - 1
            eta_seconds = _estimate_processing_time()
            return {
                "status": "queued",
                "batch_id": batch_id,
                "message": f"Query added to processing queue (position: {queue_position})",
                "position": queue_position,
                "eta_seconds": eta_seconds
            }
            
    except ValidationError as e:
        # Handle validation errors with structured logging
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except ModelProviderError as e:
        # Handle model provider errors
        error = ProcessingError(
            message=f"Model provider error: {str(e)}",
            operation="query_processing",
            resource=request.query,
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )
    except ModelConfigurationError as e:
        # Handle configuration errors
        error = ConfigurationError(
            message=f"Model configuration error: {str(e)}",
            config_key="model_config",
            config_value=str(e)
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )
    except Exception as e:
        # Handle all other exceptions
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        error = ProcessingError(
            message="Error processing query",
            operation="process_query",
            resource=request.query,
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=500, 
            detail=error.to_dict()
        )

@router.post("/batch_process")
async def batch_process(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    agent: KnowledgeAgent = Depends(get_agent)) -> Dict[str, Any]:
    """Process multiple queries in the background with proper agent lifecycle."""
    start_time = time.time()
    try:
        # Generate unique task ID
        task_id = f"batch_{int(time.time())}_{os.getpid()}"
        async with _tasks_lock:
            _background_tasks[task_id] = {
                "status": "initializing",
                "start_time": datetime.now(pytz.UTC),
                "total_queries": len(request.queries),
                "completed_queries": 0,
                "results": [],
                "errors": []}
        # Define background processing function
        async def process_batch():
            batch_start_time = time.time()
            try:
                config = await build_unified_config(request.config)
                results = []
                errors = []
                for query in request.queries:
                    query_start_time = time.time()
                    try:
                        result = await _run_inference_async(
                            query=query,
                            agent=agent,
                            config=config)
                        results.append({
                            "query": query,
                            "result": result,
                            "duration_ms": round((time.time() - query_start_time) * 1000, 2)})
                        async with _tasks_lock:
                            _background_tasks[task_id]["completed_queries"] += 1
                            _background_tasks[task_id]["results"] = results
                            
                    except Exception as query_error:
                        error = ProcessingError(
                            message=f"Error processing query: {str(query_error)}",
                            operation="process_single_query",
                            resource=query,
                            original_error=query_error
                        )
                        error.log_error(logger)
                        errors.append(error.to_dict())
                        results.append({
                            "query": query,
                            "error": error.to_dict(),
                            "duration_ms": round((time.time() - query_start_time) * 1000, 2)
                        })
                
                async with _tasks_lock:
                    _background_tasks[task_id].update({
                        "status": "completed",
                        "end_time": datetime.now(pytz.UTC),
                        "duration_ms": round((time.time() - batch_start_time) * 1000, 2),
                        "results": results,
                        "errors": errors
                    })
                
                # Log completion
                log_endpoint_call(
                    logger=logger,
                    endpoint=f"/batch_process/{task_id}",
                    method="BACKGROUND",
                    duration_ms=round((time.time() - batch_start_time) * 1000, 2),
                    params={
                        "total_queries": len(request.queries),
                        "completed_queries": len(results),
                        "error_count": len(errors)
                    }
                )
                
                # Ensure proper cleanup
                gc.collect()
                
            except Exception as batch_error:
                error = ProcessingError(
                    message="Batch processing failed",
                    operation="process_batch",
                    resource=task_id,
                    original_error=batch_error
                )
                error.log_error(logger)
                async with _tasks_lock:
                    _background_tasks[task_id].update({
                        "status": "failed",
                        "end_time": datetime.now(pytz.UTC),
                        "error": error.to_dict(),
                        "duration_ms": round((time.time() - batch_start_time) * 1000, 2)
                    })
        
        # Add task to background tasks
        background_tasks.add_task(process_batch)
        
        response = {
            "status": "accepted",
            "message": "Batch processing started",
            "task_id": task_id,
            "status_endpoint": f"/batch_status/{task_id}"
        }

        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_endpoint_call(
            logger=logger,
            endpoint="/batch_process",
            method="POST",
            duration_ms=duration_ms,
            params={
                "task_id": task_id,
                "total_queries": len(request.queries)
            }
        )
        
        return response
        
    except ValidationError as e:
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
        error = ProcessingError(
            message="Failed to start batch processing",
            operation="initialize_batch",
            resource=str(request.queries),
            original_error=e
        )
        error.log_error(logger)
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_dict()
        )

@router.get("/batch_status/{task_id}")
async def get_batch_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a batch processing task.
    
    This endpoint handles both:
    1. Multi-query batch tasks (task_id format: batch_{timestamp}_{pid})
    2. Single-query batch tasks (task_id format: batch_{timestamp}_{queue_length})
    """
    start_time = time.time()
    try:
        # First check background tasks (multi-query batches)
        if task_id in _background_tasks:
            task_status = _background_tasks[task_id]
            response = {
                "status": task_status["status"],
                "total": task_status["total_queries"],
                "completed": task_status["completed_queries"],
                "results": task_status.get("results", []),
                "errors": task_status.get("errors", []),
                "duration_ms": task_status.get("duration_ms")
            }
        else:
            # Check single-query batch results
            if task_id in _batch_results:
                result = _batch_results[task_id]
                # Don't delete results immediately to prevent "not found" errors
                # Keep them for at least 5 minutes
                response = {
                    "status": "completed",
                    "result": result
                }
            else:
                # Check if query is still in queue
                position = None
                for i, item in enumerate(_query_batch):
                    if item["id"] == task_id:
                        position = i
                        break
                
                if position is not None:
                    response = {
                        "status": "queued",
                        "position": position,
                        "eta_seconds": _estimate_processing_time()
                    }
                else:
                    # Check if the result is in the cache
                    cache_key = f"query_result:{task_id}"
                    cached_result = await cache.get(cache_key)
                    if cached_result:
                        response = {
                            "status": "completed",
                            "result": cached_result
                        }
                    else:
                        # The task ID might be valid but processing hasn't completed yet
                        # Check if it matches our expected batch ID format
                        if task_id.startswith("batch_"):
                            try:
                                # Extract timestamp from batch ID
                                timestamp_str = task_id.split("_")[1]
                                timestamp = int(timestamp_str)
                                current_time = int(time.time())
                                
                                # Extend the timeout period to 10 minutes (600 seconds) for better resilience
                                # Previously was only 60 seconds which was too short
                                if current_time - timestamp < 600:
                                    # Check if this might be a task in progress by checking embeddings status
                                    is_embedding_active = False
                                    try:
                                        async with _tasks_lock:
                                            if _embedding_task_key in _background_tasks and _background_tasks[_embedding_task_key]["status"] == "running":
                                                is_embedding_active = True
                                    except Exception:
                                        pass
                                        
                                    if is_embedding_active:
                                        response = {
                                            "status": "preprocessing",
                                            "message": "The system is currently generating embeddings, which may affect task processing. Please try again in a few minutes.",
                                            "estimated_wait_time": 300  # 5 minutes
                                        }
                                    else:
                                        response = {
                                            "status": "processing",
                                            "message": "The task is being processed. Please try again in a few seconds.",
                                            "task_id": task_id,
                                            "created_at": datetime.fromtimestamp(timestamp, tz=pytz.UTC).isoformat()
                                        }
                                else:
                                    # Even for old task IDs, first try to determine if it ever existed
                                    # Check persistent storage if available
                                    try:
                                        # Try checking a history file for evidence of this task
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
                                            except Exception as e:
                                                logger.warning(f"Error reading batch history: {e}")
                                    except Exception as e:
                                        logger.warning(f"Error checking task history: {e}")
                                    
                                    # No record found, task truly doesn't exist or is too old
                                    raise ValidationError(
                                        message=f"Task {task_id} not found or has expired",
                                        field="task_id",
                                        value=task_id
                                    )
                            except (IndexError, ValueError):
                                raise ValidationError(
                                    message=f"Invalid task ID format: {task_id}",
                                    field="task_id",
                                    value=task_id
                                )
                        else:
                            raise ValidationError(
                                message=f"Task {task_id} not found or has invalid format",
                                field="task_id",
                                value=task_id
                            )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_endpoint_call(
            logger=logger,
            endpoint=f"/batch_status/{task_id}",
            method="GET",
            duration_ms=duration_ms,
            params={"task_id": task_id}
        )
        
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
    select_board: Optional[str] = None
):
    """Process a query using recent data from the last 6 hours with batch processing.
    
    This endpoint aligns with process_query by using the same batch processing system,
    ensuring consistent behavior and response formats across both endpoints.
    
    Args:
        background_tasks: For background processing
        agent: Knowledge agent for processing 
        data_ops: Data operations for preparation
        select_board: Optional board ID to filter data
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
                config_value=str(stored_queries_path)
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                message="Invalid YAML format in stored queries",
                config_key="stored_queries",
                original_error=e
            )

        # Create request object with consistent parameters
        request = QueryRequest(
            query=query,
            filter_date=filter_date,
            force_refresh=False,
            skip_embeddings=True,
            skip_batching=False,  # Important: Use batch processing like process_query
            select_board=select_board  # Add the select_board parameter
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
            params=log_params,
            duration_ms=None
        )

        # Reuse process_query endpoint logic for consistency
        result = await process_query(
            request=request,
            agent=agent,
            data_ops=data_ops,
            background_tasks=background_tasks
        )

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
        hits = CACHE_HITS.value
        misses = CACHE_MISSES.value
        errors = CACHE_ERRORS.value
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

# Utility function for structured logging
def log_endpoint_call(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None
):
    """Log endpoint calls with consistent structure."""
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }
    if params:
        log_data["parameters"] = params
    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    logger.info(
        f"{method} {endpoint}",
        extra={"endpoint_data": log_data}
    )

async def _add_to_query_batch(request: QueryRequest) -> str:
    """Add query to batch queue and return batch ID."""
    batch_id = f"batch_{int(time.time())}_{len(_query_batch)}"
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
    
    This function:
    1. Ensures that necessary data is loaded and ready for inference
    2. Handles data refresh based on configuration parameters
    3. Manages embedding generation if needed
    
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
        
        # Check if data needs refreshing based on config and force flag
        data_freshness_check = await data_ops.check_data_freshness()
        
        # Only refresh if force_refresh is True or data is actually not fresh
        # Critical fix: ensure force_refresh is properly respected
        if force_refresh:
            logger.info("Force refresh requested, ensuring data ready with full refresh")
            await data_ops.ensure_data_ready(force_refresh=True)
        else:
            logger.info("Using existing data (force_refresh=false), ensuring minimal data preparation")
            # Pass force_refresh=False to ensure_data_ready to respect the flag
            await data_ops.ensure_data_ready(force_refresh=False)
        
        # Load stratified data
        stratified_data = await data_ops._load_stratified_data()
        
        # Check data validity
        if stratified_data is None or stratified_data.empty:
            raise ProcessingError(
                message="Stratified data is empty or not available",
                operation="load_stratified_data"
            )
        
        data_rows = len(stratified_data)
        logger.info(f"Data preparation complete: {data_rows} rows available " +
                   f"(took {round((time.time() - start_time) * 1000, 2)}ms)")
        
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
        logger.error(f"Error preparing data: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to provide more context based on exception type
        if "embedding" in str(e).lower():
            error_message = f"Embedding generation failed: {str(e)}"
            operation = "embedding_generation"
        elif "stratif" in str(e).lower():
            error_message = f"Data stratification failed: {str(e)}"
            operation = "data_stratification"
        else:
            error_message = f"Error preparing data: {str(e)}"
            operation = "data_preparation"
            
        raise ProcessingError(
            message=error_message,
            operation=operation,
            original_error=e
        )

async def _process_query_batch(
    agent: KnowledgeAgent,
    data_ops: DataOperations
) -> Dict[str, Any]:
    """Process a batch of queries using parallel processing with enhanced reliability.
    
    This function:
    1. Extracts a batch of queries from the queue with thread safety
    2. Prepares data once for all queries (optimization)
    3. Processes queries in parallel with individual error handling
    4. Stores and caches results with proper error recovery
    5. Updates timing metrics for future ETA calculations
    
    Args:
        agent: Knowledge agent for processing
        data_ops: Data operations for data preparation
        
    Returns:
        Dictionary of batch results
    """
    global _last_batch_processing_time
    
    batch_start_time = time.time()
    batch_size = 0
    batch = []
    batch_ids = []
    
    try:
        # Get batch of queries with proper locking
        async with _query_batch_lock:
            while _query_batch and len(batch) < _BATCH_SIZE:
                item = _query_batch.popleft()
                batch.append(item["request"])
                batch_ids.append(item["id"])

        batch_size = len(batch)
        if batch_size == 0:
            logger.info("No queries in batch to process")
            return {}

        logger.info(f"Processing batch of {batch_size} queries")
        
        # Track batch processing in history
        batch_history_updates = {
            batch_id: {
                "status": "processing",
                "created_at": datetime.now(pytz.UTC).isoformat(),
                "query": str(request.query)[:100] if hasattr(request, 'query') else "Unknown"
            }
            for batch_id, request in zip(batch_ids, batch)
        }
        await _update_batch_history(batch_history_updates)

        # Build unified config from first request (optimization for similar requests)
        config = build_unified_config(batch[0])
        
        # Extract providers from config for all model operations (reused across queries)
        providers = {
            ModelOperation.EMBEDDING: config.get_provider(ModelOperation.EMBEDDING),
            ModelOperation.CHUNK_GENERATION: config.get_provider(ModelOperation.CHUNK_GENERATION),
            ModelOperation.SUMMARIZATION: config.get_provider(ModelOperation.SUMMARIZATION)
        }
        
        # Process requests in groups based on their force_refresh flag
        refresh_groups = {True: [], False: []}  # Simplified dictionary initialization
        
        # Group requests by their force_refresh value
        for i, req in enumerate(batch):
            refresh_groups[req.force_refresh].append((i, req))
        
        results = [None] * len(batch)  # Pre-allocate results list
        
        # Process each group separately
        for force_refresh, group in refresh_groups.items():
            if not group:
                continue
                
            indices, group_requests = zip(*group)
            
            try:
                # First handle the special case of force_refresh=False with missing embeddings
                if not force_refresh:
                    await _handle_missing_embeddings(data_ops)
                
                # Prepare data with the appropriate force_refresh flag for this group
                stratified_data = await _prepare_data_if_needed(
                    config=config,
                    data_ops=data_ops,
                    force_refresh=force_refresh,
                    skip_embeddings=all(req.skip_embeddings for req in group_requests)
                )
                
                # Process this group's queries
                group_results = await process_multiple_queries(
                    queries=[req.query for req in group_requests],
                    agent=agent,
                    stratified_data=stratified_data,
                    chunk_batch_size=config.get_batch_size(ModelOperation.CHUNK_GENERATION),
                    summary_batch_size=config.get_batch_size(ModelOperation.SUMMARIZATION),
                    providers=providers
                )
                
                # Store results in the correct positions
                for idx, result in zip(indices, group_results):
                    results[idx] = result
                    
            except Exception as e:
                logger.error(f"Error processing {force_refresh=} group: {str(e)}")
                # Mark all queries in this group as failed
                for idx, _ in group:
                    results[idx] = (None, str(e))
                    _batch_results[batch_ids[idx]] = {
                        "error": str(e),
                        "status": "failed",
                        "timestamp": time.time()
                    }
                    batch_history_updates[batch_ids[idx]] = {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now(pytz.UTC).isoformat()
                    }
        
        # Check if any results are None (indicating complete failure of a group)
        if any(r is None for r in results):
            failed_indices = [i for i, r in enumerate(results) if r is None]
            raise ProcessingError(
                message=f"Batch processing failed for indices: {failed_indices}",
                operation="process_multiple_queries"
            )

        logger.info(f"Successfully processed {len(results)} queries in batch")
        
        # Store results with improved error handling for individual results
        store_tasks = [
            _store_batch_result(batch_id, result, config) 
            for batch_id, result in zip(batch_ids, results)
        ]
        storage_results = await asyncio.gather(*store_tasks, return_exceptions=True)
        successful_stores = sum(1 for result in storage_results if result is True)
        logger.info(f"Successfully stored {successful_stores}/{len(storage_results)} results")

        # Schedule cleanup of old results (but don't wait for it)
        asyncio.create_task(_cleanup_old_batch_results())

        # Update timing metrics for ETA calculations
        _last_batch_processing_time = time.time() - batch_start_time
        
        # Log batch processing statistics
        logger.info(
            f"Processed batch of {batch_size} queries in {_last_batch_processing_time:.2f}s " +
            f"({_last_batch_processing_time/batch_size:.2f}s per query)"
        )
        
        # Update batch history with completed tasks
        await _update_batch_history(batch_history_updates)
        
        # Clean up to free memory
        gc.collect()
        
        return _batch_results
        
    except Exception as e:
        batch_processing_time = time.time() - batch_start_time
        logger.error(f"Error processing batch: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Mark all affected queries as failed with detailed error info
        batch_history_updates = {
            batch_id: {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "failed_at": datetime.now(pytz.UTC).isoformat()
            }
            for batch_id in batch_ids
        }
        
        # Record failures in batch results
        for batch_id in batch_ids:
            _batch_results[batch_id] = {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                },
                "status": "failed",
                "timestamp": time.time(),
                "processing_time": batch_processing_time
            }
        
        # Update batch history with failed tasks
        if batch_history_updates:
            await _update_batch_history(batch_history_updates)
        
        # Try to free memory after error
        gc.collect()
        
        # Propagate the error for proper handling
        if isinstance(e, (APIError, ModelProviderError, ModelConfigurationError)):
            raise
        else:
            raise ProcessingError(
                message=f"Batch processing failed: {str(e)}",
                operation="process_batch",
                resource=f"batch_size_{batch_size}",
                original_error=e
            )

async def _handle_missing_embeddings(data_ops: DataOperations) -> None:
    """Handle the case of missing embeddings without full refresh."""
    try:
        stratified_file = data_ops.config.stratified_data_path / 'stratified_sample.csv'
        embeddings_path = data_ops.config.stratified_data_path / 'embeddings.npz'
        thread_id_map_path = data_ops.config.stratified_data_path / 'thread_id_map.json'
        
        if all(path.exists() for path in [stratified_file, embeddings_path, thread_id_map_path]):
            stratified_data = await data_ops._load_stratified_data()
            if stratified_data is not None and len(stratified_data) > 0:
                with np.load(embeddings_path) as data:
                    embeddings = data.get('embeddings')
                with open(thread_id_map_path, 'r') as f:
                    thread_id_map = json.load(f)
                
                embedding_count = len(embeddings) if embeddings is not None else 0
                stratified_count = len(stratified_data)
                
                if 0 < embedding_count < stratified_count:
                    logger.info(f"Embedding mismatch: {embedding_count} embeddings for {stratified_count} records")
                    await data_ops.generate_missing_embeddings(stratified_data, thread_id_map)
                    logger.info("Generated missing embeddings without full refresh")
    except Exception as e:
        logger.warning(f"Error checking for embedding mismatch: {e}")

async def _store_batch_result(batch_id: str, result_tuple: tuple, config: ModelConfig) -> bool:
    """Store an individual query result with error handling."""
    try:
        chunks, summary = result_tuple
        result = {
            "chunks": chunks,
            "summary": summary,
            "config": {
                "filter_date": config.processing_settings.get('filter_date'),
                "sample_size": config.processing_settings.get('sample_size'),
                "max_workers": config.processing_settings.get('max_workers')
            },
            "timestamp": time.time()
        }
        
        # Store in batch results
        _batch_results[batch_id] = result
        
        # Also cache the result
        cache_key = f"query_result:{batch_id}"
        await cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour
        
        # Update batch history
        batch_history_updates = {
            batch_id: {
                "status": "completed",
                "completed_at": datetime.now(pytz.UTC).isoformat(),
                "summary_length": len(summary) if summary else 0,
                "chunks_count": len(chunks) if chunks else 0
            }
        }
        await _update_batch_history(batch_history_updates)
        return True
    except Exception as e:
        logger.error(f"Error storing result for batch {batch_id}: {str(e)}")
        _batch_results[batch_id] = {
            "error": str(e),
            "status": "failed",
            "timestamp": time.time()
        }
        batch_history_updates = {
            batch_id: {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now(pytz.UTC).isoformat()
            }
        }
        await _update_batch_history(batch_history_updates)
        return False

async def _cleanup_old_batch_results():
    """Clean up old batch results that are no longer needed.
    
    This function:
    1. Identifies and removes batch results older than the retention period
    2. Updates the batch history with completion information
    3. Ensures proper resource cleanup to prevent memory leaks
    
    The cleanup is performed asynchronously and doesn't block the main processing flow.
    Results are kept for 5 minutes by default to allow clients time to retrieve them.
    """
    try:
        # Define retention period (5 minutes = 300 seconds)
        retention_period = int(os.getenv('BATCH_RESULT_RETENTION_SECONDS', '300'))
        current_time = time.time()
        batch_ids_to_remove = []
        batch_history_updates = {}
        
        # Identify old results to clean up
        for batch_id, result in _batch_results.items():
            # Check if result has a timestamp
            timestamp = result.get("timestamp", 0)
            if current_time - timestamp > retention_period:
                batch_ids_to_remove.append(batch_id)
                # Save completion info to history
                status = "completed" if "error" not in result else "failed"
                batch_history_updates[batch_id] = {
                    "status": status,
                    "completed_at": datetime.fromtimestamp(timestamp, tz=pytz.UTC).isoformat(),
                    "removed_at": datetime.now(pytz.UTC).isoformat(),
                    "retention_period_seconds": retention_period
                }
        
        # Remove old results if any found
        if batch_ids_to_remove:
            # Use a lock to avoid race conditions if needed
            for batch_id in batch_ids_to_remove:
                if batch_id in _batch_results:
                    # Get result size before removal for logging
                    try:
                        result_size = len(str(_batch_results[batch_id]))
                    except Exception:
                        result_size = 0
                    
                    # Remove from cache and memory
                    del _batch_results[batch_id]
                    # Also try to remove from the cache
                    try:
                        cache_key = f"query_result:{batch_id}"
                        asyncio.create_task(cache.delete(cache_key))
                    except Exception as cache_error:
                        logger.debug(f"Cache removal error for {batch_id}: {str(cache_error)}")
                        
            # Log cleanup stats
            logger.info(f"Cleaned up {len(batch_ids_to_remove)} old batch results (retention: {retention_period}s)")
            
            # Update batch history file with completed tasks
            if batch_history_updates:
                await _update_batch_history(batch_history_updates)
                
            # Force garbage collection after large cleanups
            if len(batch_ids_to_remove) > 10:
                gc.collect()
                
    except Exception as e:
        # Don't let cleanup failures affect the main process
        logger.error(f"Error during batch result cleanup: {str(e)}")
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
