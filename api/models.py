"""API response models and logging utilities.

This module contains models and utilities specific to the API layer:
1. Response Models: Pydantic models for API responses (e.g., HealthResponse)
2. Logging Utilities: Consistent logging for API endpoints

Note: This module is intentionally separate from:
- config/config_utils.py: Which handles configuration models
- knowledge_agents/model_ops.py: Which handles ML model operations

This separation ensures clear boundaries between API responses,
configuration management, and ML operations.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    timestamp: datetime
    environment: Dict[str, Any]

class StratificationResponse(BaseModel):
    """Response model for data stratification endpoint."""
    status: str
    message: str
    timestamp: datetime
    total_records: int
    stratified_records: int
    sample_size: int
    filter_date: Optional[str] = None
    stratification_details: Dict[str, Any]

class QueryResponse(BaseModel):
    """Response model for query endpoint.
    
    Handles both synchronous and asynchronous query responses.
    For background processing, chunks/summary/metadata may be empty.
    """
    status: str
    task_id: str
    chunks: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class BatchProcessResponse(BaseModel):
    """Response model for batch processing endpoint."""
    batch_id: str
    results: List[Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

class BatchStatusResponse(BaseModel):
    """Response model for batch status endpoint."""
    status: str
    total: Optional[int] = None
    completed: Optional[int] = None
    results: Optional[List[Any]] = None
    errors: Optional[List[str]] = None
    duration_ms: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    position: Optional[int] = None
    eta_seconds: Optional[int] = None
    message: Optional[str] = None
    task_id: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    estimated_wait_time: Optional[int] = None

class EmbeddingStatusResponse(BaseModel):
    """Response model for embedding status endpoint."""
    status: str
    message: str
    progress: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    total_rows: int = 0

def log_endpoint_call(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    duration_ms: float,
    params: Optional[Dict[str, Any]] = None
) -> None:
    """Log API endpoint calls with consistent format.
    
    Args:
        logger: Logger instance to use
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        duration_ms: Request duration in milliseconds
        params: Optional parameters to log
    """
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if params:
        log_data["params"] = params
    
    logger.info(
        f"{method} {endpoint} completed in {duration_ms}ms",
        extra={"request_data": log_data}
    ) 