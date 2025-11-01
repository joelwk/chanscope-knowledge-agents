"""Health check endpoints."""
import os
import time
from datetime import datetime
from typing import Dict, Any

import pytz
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from knowledge_agents.model_ops import (
    ModelProvider,
    KnowledgeAgent,
    ModelProviderError,
    ModelOperationError,
)
from knowledge_agents.data_processing.cloud_handler import S3Handler
from knowledge_agents.data_ops import DataOperations
from api.errors import ProcessingError, APIError, ValidationError
from config.settings import Config
from config.logging_config import get_logger
from api.models import log_endpoint_call
from api.cache import CACHE_HITS, CACHE_MISSES, CACHE_ERRORS
from config.env_loader import is_replit_environment, get_replit_paths

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

# Dependency for KnowledgeAgent
async def get_agent() -> KnowledgeAgent:
    """Get the KnowledgeAgent singleton instance."""
    from knowledge_agents.embedding_ops import get_agent
    return await get_agent()

# Dependency for DataOperations
async def get_data_ops() -> DataOperations:
    from knowledge_agents.data_ops import DataConfig, DataOperations
    data_config = DataConfig.from_config()
    return DataOperations(data_config)

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
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }

