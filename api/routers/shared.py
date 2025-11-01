"""Shared utilities for API routers.

This module contains shared state and helper functions used across multiple routers.
"""
import asyncio
import secrets
import re
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import traceback

from knowledge_agents.model_ops import ModelConfig
from knowledge_agents.data_ops import DataOperations
from config.settings import Config
from config.logging_config import get_logger
from api.cache import cache

logger = get_logger(__name__)

# Global variables for tracking tasks and results
_background_tasks: Dict[str, Dict[str, Any]] = {}
_batch_results: Dict[str, Any] = {}
_query_batch: List[Dict[str, Any]] = []
_BATCH_SIZE = Config.get_model_settings()['embedding_batch_size']
_BATCH_WAIT_TIME = 2  # Seconds to wait for batch accumulation
_last_batch_processing_time = 0  # Track processing time for ETA estimates
_tasks_lock = asyncio.Lock()
_embedding_task_key = "embedding_generation"

task_response_queue: asyncio.Queue = asyncio.Queue()

# Getter functions for accessing shared state
def get_background_tasks() -> Dict[str, Dict[str, Any]]:
    """Get the global background tasks dictionary."""
    return _background_tasks

def get_batch_results() -> Dict[str, Any]:
    """Get the global batch results dictionary."""
    return _batch_results

def get_query_batch() -> List[Dict[str, Any]]:
    """Get the global query batch list."""
    return _query_batch

def get_tasks_lock() -> asyncio.Lock:
    """Get the global tasks lock."""
    return _tasks_lock

def get_embedding_task_key() -> str:
    """Get the embedding task key."""
    return _embedding_task_key

def get_batch_size() -> int:
    """Get the batch size."""
    return _BATCH_SIZE

def get_batch_wait_time() -> int:
    """Get the batch wait time."""
    return _BATCH_WAIT_TIME

def get_last_batch_processing_time() -> float:
    """Get the last batch processing time."""
    return _last_batch_processing_time

def set_last_batch_processing_time(time_ms: float) -> None:
    """Set the last batch processing time."""
    global _last_batch_processing_time
    _last_batch_processing_time = time_ms

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

def _estimate_processing_time() -> int:
    """Estimate processing time based on batch size and historical data."""
    batch_size = len(_query_batch)
    avg_time_per_query = _last_batch_processing_time / _BATCH_SIZE if _last_batch_processing_time > 0 else 2
    position_in_queue = len(_query_batch) - 1
    return int(avg_time_per_query * (position_in_queue // _BATCH_SIZE + 1))

async def _store_batch_result(
    batch_id: str, 
    result: Union[tuple, Dict[str, Any]], 
    config: Union[Dict[str, Any], ModelConfig], 
    save_to_disk: bool = True
) -> bool:
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
        # Early detection guard for future contract drift
        if not isinstance(result, (tuple, dict)):
            logger.warning(f"Unexpected result type detected in _store_batch_result: {type(result)}. Expected tuple or dict. Value: {result}")
            # Continue processing but log the issue for investigation
        
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

async def _update_batch_history(updates: Dict[str, Dict[str, Any]]) -> None:
    """Update the batch history file with task status information.

    This maintains a record of all batch tasks even after they're removed from memory.

    Args:
        updates: Dictionary mapping batch IDs to their status information
    """
    try:
        from filelock import FileLock
        import shutil
        
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

async def _prepare_data_if_needed(
    config: ModelConfig,
    data_ops: DataOperations,
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> Any:
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
    import pandas as pd
    from fastapi import HTTPException
    from config.env_loader import detect_environment
    
    # Check environment type to use appropriate data readiness methods
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
            from config.settings import Config
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

# Export task ID utilities and helper functions
generate_task_id = _generate_task_id
is_valid_task_id = _is_valid_task_id
check_task_id_collision = _check_task_id_collision
estimate_processing_time = _estimate_processing_time
store_batch_result = _store_batch_result
update_batch_history = _update_batch_history
prepare_data_if_needed = _prepare_data_if_needed
cleanup_old_results = _cleanup_old_results

