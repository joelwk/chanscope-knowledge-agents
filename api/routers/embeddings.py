"""Embedding generation endpoints."""
from datetime import datetime
from typing import Dict, Any

import pytz
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from knowledge_agents.data_ops import DataOperations
from api.errors import ProcessingError
from api.models import EmbeddingStatusResponse
from api.routers.shared import (
    get_background_tasks,
    get_tasks_lock,
    get_embedding_task_key
)
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["embeddings"])

# Dependency for DataOperations
async def get_data_ops() -> DataOperations:
    from knowledge_agents.data_ops import DataConfig, DataOperations
    data_config = DataConfig.from_config()
    return DataOperations(data_config)

async def _run_embedding_generation(data_ops: DataOperations) -> None:
    """Background task for generating embeddings.

    This function handles the embedding generation process with proper state tracking
    and error handling.

    Args:
        data_ops: DataOperations instance for data processing
    """
    import traceback
    from api.routers.shared import get_background_tasks, get_tasks_lock, get_embedding_task_key
    
    _background_tasks = get_background_tasks()
    _tasks_lock = get_tasks_lock()
    _embedding_task_key = get_embedding_task_key()
    
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

        async def progress_callback(progress: int, total: int) -> None:
            """Update embedding generation progress."""
            _background_tasks = get_background_tasks()
            _tasks_lock = get_tasks_lock()
            _embedding_task_key = get_embedding_task_key()
            async with _tasks_lock:
                if _embedding_task_key in _background_tasks:
                    progress_pct = min(100, int((progress / total) * 100))
                    _background_tasks[_embedding_task_key]['progress'] = progress_pct
                    logger.info(f"Embedding generation progress: {progress_pct}%")

        await data_ops._update_embeddings(
            stratified_data=df,
            progress_callback=lambda current: progress_callback(current, total_rows)
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

@router.post("/trigger_embedding_generation")
async def trigger_embedding_generation(
    background_tasks: BackgroundTasks,
    data_ops: DataOperations = Depends(get_data_ops)
):
    """Trigger background embedding generation process."""
    _background_tasks = get_background_tasks()
    _tasks_lock = get_tasks_lock()
    _embedding_task_key = get_embedding_task_key()
    
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

@router.get("/embedding_status", response_model=EmbeddingStatusResponse)
async def get_embedding_status() -> EmbeddingStatusResponse:
    """Get detailed status of background embedding generation."""
    _background_tasks = get_background_tasks()
    _tasks_lock = get_tasks_lock()
    _embedding_task_key = get_embedding_task_key()
    
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

