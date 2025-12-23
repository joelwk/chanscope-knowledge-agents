"""Query processing endpoints."""
import time
import traceback
import os
import yaml
import asyncio
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import json
import copy
import random

import pytz
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from knowledge_agents.model_ops import (
    ModelProvider,
    KnowledgeAgent,
    ModelProviderError,
    ModelConfigurationError,
    ModelOperationError,
    ModelOperation,
    ModelConfig,
    load_config
)
from knowledge_agents.data_ops import DataConfig, DataOperations
from knowledge_agents.inference_ops import process_multiple_queries_efficient, process_query
from api.errors import ProcessingError, APIError, ConfigurationError, ValidationError

from config.base_settings import get_base_settings
from config.settings import Config
from config.config_utils import (
    QueryRequest,
    BatchQueryRequest,
    build_unified_config
)
from config.logging_config import get_logger
from api.models import log_endpoint_call, QueryResponse, BatchProcessResponse, BatchStatusResponse
from api.cache import cache
from config.env_loader import is_replit_environment, detect_environment
from knowledge_agents.utils import save_query_output

# Import the LLM-based SQL generator
from knowledge_agents.llm_sql_generator import LLMSQLGenerator, NLQueryParsingError

# Import shared utilities
from api.routers.shared import (
    get_background_tasks,
    get_batch_results,
    get_query_batch,
    get_tasks_lock,
    get_embedding_task_key,
    generate_task_id,
    is_valid_task_id,
    check_task_id_collision,
    estimate_processing_time,
    store_batch_result,
    prepare_data_if_needed
)

logger = get_logger(__name__)
router = APIRouter(tags=["query"])

# Dependency for KnowledgeAgent
async def get_agent() -> KnowledgeAgent:
    """Get the KnowledgeAgent singleton instance."""
    from knowledge_agents.embedding_ops import get_agent
    return await get_agent()

# Dependency for DataOperations
async def get_data_ops() -> DataOperations:
    data_config = DataConfig.from_config()
    return DataOperations(data_config)

# Define request models for NL queries
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
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            datetime: lambda x: x.isoformat(),
            date: lambda x: x.isoformat(),
        }
        populate_by_name = True
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
                "data": [],
                "execution_time_ms": 189.72,
                "metadata": {}
            }
        }

def format_response_for_llm(response: Dict[str, Any]) -> Dict[str, Any]:
    """Format response data to be more friendly for LLM consumption."""
    formatted = copy.deepcopy(response)

    if "data" in formatted and isinstance(formatted["data"], list):
        if formatted["data"]:
            all_keys = set()
            for record in formatted["data"]:
                all_keys.update(record.keys())

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

            formatted_data = []
            for record in formatted["data"]:
                formatted_record = {}
                for key in sorted_keys:
                    if key in record:
                        formatted_record[key] = record[key]
                formatted_data.append(formatted_record)

            formatted["data"] = formatted_data

    return formatted

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
    _background_tasks = get_background_tasks()
    _tasks_lock = get_tasks_lock()
    
    logger.info(f"Processing query {task_id}: {query[:50]}...")

    try:
        async with _tasks_lock:
            _background_tasks[task_id] = {
                "status": "processing",
                "timestamp": time.time(),
                "query": query,
                "progress": 10
            }

        from config.env_loader import detect_environment
        env_type = detect_environment()

        if data_ops is None:
            data_ops = await get_data_ops()

        # Ensure data is ready based on environment
        if env_type.lower() == 'replit':
            logger.info("Using Replit-specific storage for data readiness check")
            from config.storage import StorageFactory
            storage = StorageFactory.create(data_ops.config, env_type)

            complete_data_storage = storage['complete_data']
            stratified_storage = storage['stratified_sample']

            row_count = await complete_data_storage.get_row_count()
            logger.info(f"PostgreSQL database has {row_count} rows")

            stratified_exists = await stratified_storage.sample_exists()
            logger.info(f"Stratified sample exists: {stratified_exists}")

            if not stratified_exists:
                logger.info("Stratified sample doesn't exist in Replit storage, forcing refresh")
                force_refresh = True

            data_is_ready = (row_count > 0 and stratified_exists)
            if not data_is_ready or force_refresh:
                logger.info(f"Data not ready or force_refresh={force_refresh}, preparing data...")
                await data_ops.ensure_data_ready(
                    force_refresh=force_refresh,
                    skip_embeddings=skip_embeddings
                )
        else:
            if force_refresh or not await data_ops.is_data_ready(skip_embeddings=skip_embeddings):
                logger.info(f"Data not ready or force_refresh={force_refresh}, preparing data...")
                await data_ops.ensure_data_ready(
                    force_refresh=force_refresh,
                    skip_embeddings=skip_embeddings
                )

        logger.info("Loading stratified data for background processing...")
        library_df = await data_ops.load_stratified_data()
        logger.info(f"Loaded {len(library_df)} rows from stratified data")

        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["progress"] = 30

        if use_batching:
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
            result = await agent.process_query(
                query=query,
                df=library_df,
                model=config.get_provider(ModelOperation.SUMMARIZATION)
            )

        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["progress"] = 80

        success = await store_batch_result(
            batch_id=task_id,
            result=result,
            config=config
        )

        if not success:
            logger.warning(f"Failed to store results for task {task_id}")

        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["status"] = "completed"
                _background_tasks[task_id]["progress"] = 100
                _background_tasks[task_id]["completed_at"] = time.time()

        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error processing query {task_id}: {str(e)}")
        logger.error(traceback.format_exc())

        async with _tasks_lock:
            if task_id in _background_tasks:
                _background_tasks[task_id]["status"] = "failed"
                _background_tasks[task_id]["error"] = str(e)
                _background_tasks[task_id]["progress"] = 100
                _background_tasks[task_id]["completed_at"] = time.time()

        error_result = {
            "chunks": [],
            "summary": f"Error processing query: {str(e)}",
            "metadata": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

        await store_batch_result(
            batch_id=task_id,
            result=error_result,
            config=config
        )

        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }

@router.post("/query", response_model=QueryResponse)
async def base_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """Process a single query and return results."""
    _background_tasks = get_background_tasks()
    _tasks_lock = get_tasks_lock()
    
    start_time = time.time()
    task_id = None

    try:
        if not request.query or not isinstance(request.query, str):
            raise ValidationError(
                message="Invalid query format",
                field="query",
                value=request.query
            )

        if request.task_id:
            if not is_valid_task_id(request.task_id):
                raise ValidationError(
                    message="Invalid task ID format. Task IDs must be 4-64 characters and contain only alphanumeric characters, underscores, and hyphens.",
                    field="task_id",
                    value=request.task_id
                )

            if check_task_id_collision(request.task_id):
                raise ValidationError(
                    message="Task ID already exists. Please provide a unique task ID.",
                    field="task_id",
                    value=request.task_id
                )

            task_id = request.task_id
            logger.info(f"Using user-provided task ID: {task_id}")
        else:
            task_id = generate_task_id(prefix="query")
            logger.info(f"Generated task ID: {task_id}")

        logger.info(f"Processing query request {task_id}: {request.query[:50]}...")

        agent = await get_agent()
        data_ops = await get_data_ops()
        config = build_unified_config(request)

        if request.use_background:
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
            try:
                async with _tasks_lock:
                    _background_tasks[task_id] = {
                        "status": "processing",
                        "timestamp": time.time(),
                        "query": request.query,
                        "progress": 0
                    }

                from config.env_loader import detect_environment
                env_type = detect_environment()

                if env_type.lower() == 'replit':
                    logger.info("Using Replit-specific storage for data readiness check")
                    from config.storage import StorageFactory
                    storage = StorageFactory.create(config, env_type)

                    complete_data_storage = storage['complete_data']
                    stratified_storage = storage['stratified_sample']

                    row_count = await complete_data_storage.get_row_count()
                    logger.info(f"PostgreSQL database has {row_count} rows")

                    stratified_exists = await stratified_storage.sample_exists()
                    logger.info(f"Stratified sample exists: {stratified_exists}")

                    if not stratified_exists:
                        logger.info("Stratified sample doesn't exist in Replit storage, forcing refresh")
                        request.force_refresh = True

                    data_is_ready = (row_count > 0 and stratified_exists)
                    if not data_is_ready or request.force_refresh:
                        logger.info(f"Data not ready or force_refresh={request.force_refresh}, preparing data...")
                        await data_ops.ensure_data_ready(
                            force_refresh=request.force_refresh,
                            skip_embeddings=request.skip_embeddings
                        )
                else:
                    if request.force_refresh or not await data_ops.is_data_ready(skip_embeddings=request.skip_embeddings):
                        logger.info(f"Data not ready or force_refresh={request.force_refresh}, preparing data...")
                        await data_ops.ensure_data_ready(
                            force_refresh=request.force_refresh,
                            skip_embeddings=request.skip_embeddings
                        )

                logger.info("Loading stratified data for processing...")
                library_df = await data_ops.load_stratified_data()

                logger.info(f"Loaded stratified data with {len(library_df)} rows and columns: {list(library_df.columns)}")

                if 'embedding' not in library_df.columns:
                    logger.warning("Embeddings not present in loaded data, checking if they need to be loaded separately")
                    try:
                        from config.storage import StorageFactory
                        from config.env_loader import detect_environment
                        env_type = detect_environment()
                        storage = StorageFactory.create(data_ops.config, env_type)
                        embedding_storage = storage['embeddings']

                        embeddings, thread_map = await embedding_storage.get_embeddings()
                        if embeddings is not None and thread_map is not None:
                            logger.info(f"Merging {len(embeddings)} embeddings with stratified data...")
                            library_df["embedding"] = None

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

                if "text_clean" not in library_df.columns and "content" in library_df.columns:
                    logger.info("Adding text_clean field from content field")
                    library_df["text_clean"] = library_df["content"]

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

                success = await store_batch_result(
                    batch_id=task_id,
                    result=result,
                    config=config
                )

                if not success:
                    raise ProcessingError(
                        message="Failed to store query results",
                        operation="store_result"
                    )

                async with _tasks_lock:
                    if task_id in _background_tasks:
                        _background_tasks[task_id]["status"] = "completed"
                        _background_tasks[task_id]["progress"] = 100

                response = {
                    "status": "completed",
                    "task_id": task_id,
                    "chunks": result.get("chunks", []),
                    "summary": result.get("summary", ""),
                    "metadata": result.get("metadata", {})
                }

                duration_ms = round((time.time() - start_time) * 1000, 2)
                response["metadata"]["processing_time_ms"] = duration_ms

                try:
                    base_path = Path(Config.get_paths()["generated_data"])
                    json_path, embeddings_path = await save_query_output(
                        response=response,
                        base_path=base_path,
                        logger=logger,
                        query=request.query,
                        task_id=task_id
                    )

                    if "metadata" not in response:
                        response["metadata"] = {}
                    
                    response["metadata"]["saved_files"] = {}
                    
                    if json_path:
                        response["metadata"]["saved_files"]["json"] = str(json_path)
                    
                    if embeddings_path:
                        response["metadata"]["saved_files"]["embeddings"] = str(embeddings_path)
                    
                    from config.env_loader import is_replit_environment
                    response["metadata"]["storage_type"] = "object_storage" if is_replit_environment() else "filesystem"

                except Exception as e:
                    logger.error(f"Error saving query output: {e}")

                logger.info(f"Query {task_id} processed in {duration_ms}ms")
                return response

            except Exception as e:
                logger.error(f"Error processing query {task_id}: {str(e)}")
                logger.error(traceback.format_exc())

                async with _tasks_lock:
                    if task_id in _background_tasks:
                        _background_tasks[task_id]["status"] = "failed"
                        _background_tasks[task_id]["error"] = str(e)

                error_result = {
                    "chunks": [],
                    "summary": f"Error processing query: {str(e)}",
                    "metadata": {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                }

                await store_batch_result(
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
                "task_id": task_id
            }
        )
    finally:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_endpoint_call(
            logger=logger,
            endpoint="/query",
            method="POST",
            duration_ms=duration_ms,
            params={"query": request.query[:50] + "..." if len(request.query) > 50 else request.query}
        )

@router.post("/batch_process", response_model=BatchProcessResponse)
async def batch_process_queries(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks = None
) -> BatchProcessResponse:
    """Process multiple queries in an optimized batch."""
    start_time = time.time()
    batch_id = generate_task_id(prefix="batch")
    logger.info(f"Processing batch of {len(request.queries)} queries: {batch_id}")

    try:
        agent = await get_agent()
        config = load_config()
        data_ops = DataOperations(DataConfig.from_config())

        stratified_data = await prepare_data_if_needed(
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
            stratified_data=stratified_data,
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

        duration_ms = round((time.time() - start_time) * 1000, 2)
        avg_time_per_query = round(duration_ms / len(request.queries), 2)
        logger.info(f"Batch processed {len(request.queries)} queries in {duration_ms}ms (avg: {avg_time_per_query}ms/query)")

        saved_result_info = []
        for i, (query, result) in enumerate(zip(request.queries, results)):
            result_id = f"{batch_id}_item_{i}"
            
            result_config = copy.deepcopy(config)
            result_config.query = query
            
            if isinstance(result, dict):
                chunks = result.get("chunks", [])
                summary = result.get("summary", "")
                meta_extra = result.get("metadata", {})
            else:
                chunks, summary = result[:2]
                meta_extra = result[2] if len(result) > 2 else {}
            
            result_dict = {
                "chunks": chunks,
                "summary": summary,
                "metadata": {
                    "batch_id": batch_id,
                    "item_index": i,
                    "query": query,
                    "processing_time_ms": duration_ms / len(request.queries),
                    **meta_extra
                }
            }
            
            success = await store_batch_result(
                batch_id=result_id,
                result=result_dict,
                config=result_config,
                save_to_disk=True
            )
            
            if success:
                saved_info = {
                    "result_id": result_id,
                    "query": query,
                    "saved": success
                }
                _batch_results = get_batch_results()
                if result_id in _batch_results and "metadata" in _batch_results[result_id]:
                    if "saved_files" in _batch_results[result_id]["metadata"]:
                        saved_info["file_paths"] = _batch_results[result_id]["metadata"]["saved_files"]
                
                saved_result_info.append(saved_info)

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

@router.get("/batch_status/{task_id}", response_model=BatchStatusResponse)
async def get_batch_status(task_id: str) -> BatchStatusResponse:
    """Get the status of a processing task."""
    _background_tasks = get_background_tasks()
    _batch_results = get_batch_results()
    _query_batch = get_query_batch()
    _tasks_lock = get_tasks_lock()
    _embedding_task_key = get_embedding_task_key()
    
    start_time = time.time()
    try:
        logger.debug(f"get_batch_status: Checking status for task_id: {task_id}")

        if task_id in _background_tasks:
            task_status = _background_tasks[task_id]
            logger.debug(f"Task {task_id} found in _background_tasks: {task_status}")
            raw_results = task_status.get("results")
            if raw_results is None:
                normalized_results = []
            elif isinstance(raw_results, list):
                normalized_results = raw_results
            else:
                normalized_results = [raw_results]
            response = {
                "status": task_status["status"],
                "total": task_status.get("total_queries"),
                "completed": task_status.get("completed_queries"),
                "results": normalized_results,
                "errors": task_status.get("errors", []),
                "duration_ms": task_status.get("duration_ms")
            }
        elif task_id in _batch_results:
            result = _batch_results[task_id]
            logger.debug(f"Task {task_id} found in _batch_results: {result}")
            response = {
                "status": "completed",
                "result": result
            }
        else:
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
                    "eta_seconds": estimate_processing_time()
                }
            else:
                cache_key = f"task_result:{task_id}"
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Task {task_id} retrieved from cache.")
                    response = {
                        "status": "completed",
                        "result": cached_result
                    }
                else:
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
                            is_embedding_active = False
                            async with _tasks_lock:
                                if _embedding_task_key in _background_tasks and _background_tasks[_embedding_task_key]["status"] == "running":
                                    is_embedding_active = True
                            if is_embedding_active:
                                logger.debug(f"Embedding generation is active for task {task_id}")
                                response = {
                                    "status": "preprocessing",
                                    "message": "Embeddings are currently being generated. Please try again shortly.",
                                    "estimated_wait_time": 300
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
    """Process a query using recent data from the last 6 hours with batch processing."""
    try:
        from config.env_loader import detect_environment
        env_type = detect_environment()

        if env_type.lower() == 'replit':
            from config.storage import StorageFactory
            storage = StorageFactory.create(data_ops.config, env_type)
            stratified_storage = storage['stratified_sample']

            stratified_exists = await stratified_storage.sample_exists()
            if not stratified_exists:
                logger.info("Stratified sample doesn't exist in Replit storage, forcing refresh")
                force_refresh = True

        end_time = datetime.now(pytz.UTC)

        try:
            start_time = end_time - timedelta(hours=12)
            logger.info(f"Using default 6-hour window: {start_time.isoformat()} to {end_time.isoformat()}")
        except Exception as e:
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

        base_settings = get_base_settings()
        paths = base_settings.get('paths', {})
        stored_queries_path = Path(paths.get('config', 'config')) / 'stored_queries.yaml'

        if env_type.lower() == 'replit':
            logger.info("Using Replit environment, loading query from configuration")
            try:
                with open(stored_queries_path, 'r') as f:
                    stored_queries = yaml.safe_load(f)
                
                if stored_queries and 'query' in stored_queries:
                    if select_board == "biz" and 'biz' in stored_queries['query'] and 'queries' in stored_queries['query']['biz']:
                        biz_queries = stored_queries['query']['biz']['queries']
                        if biz_queries:
                            query = random.choice(biz_queries)
                            logger.info(f"Using random biz query: {query}")
                        else:
                            query = "Stocks, Defi, financial market impact and cryptocurrency developments <Bitcoin, Ethereum, Solana, Chainlink, Base, MSTR, NASDAQ, S&P 500, Dow Jones, and other financial markets>"
                            logger.info(f"Using default biz query (no queries found): {query}")
                    
                    elif select_board == "pol" and 'pol' in stored_queries['query'] and 'queries' in stored_queries['query']['pol']:
                        pol_queries = stored_queries['query']['pol']['queries']
                        if pol_queries:
                            query = random.choice(pol_queries)
                            logger.info(f"Using random pol query: {query}")
                        else:
                            query = "Current events, election results, executive orders, legislation, geopolitical developments, and regulatory developments"
                            logger.info(f"Using default pol query (no queries found): {query}")
                    
                    else:
                        all_queries = []
                        if 'biz' in stored_queries['query'] and 'queries' in stored_queries['query']['biz']:
                            all_queries.extend(stored_queries['query']['biz']['queries'])
                        if 'pol' in stored_queries['query'] and 'queries' in stored_queries['query']['pol']:
                            all_queries.extend(stored_queries['query']['pol']['queries'])
                        
                        if all_queries:
                            query = random.choice(all_queries)
                            logger.info(f"Using random query (no board specified): {query}")
                        else:
                            query = "Current geopolitical events and financial market developments"
                            logger.info(f"Using default query (no queries found): {query}")
                else:
                    if select_board == "biz":
                        query = "Stocks, Defi, financial market impact and cryptocurrency developments <Bitcoin, Ethereum, Solana, Chainlink, Base, MSTR, NASDAQ, S&P 500, Dow Jones, and other financial markets>"
                    elif select_board == "pol":
                        query = "Current events, election results, executive orders, legislation, geopolitical developments, and regulatory developments"
                    else:
                        query = "Current geopolitical events and financial market developments"
                    logger.info(f"Using default query (malformed stored_queries): {query}")
            except FileNotFoundError:
                query = "Current events with financial market impact and cryptocurrency developments"
                logger.info(f"Using default query (file not found): {query}")
            except Exception as e:
                logger.warning(f"Error loading stored queries: {e}. Using default query.")
                query = "Current events with financial market impact and cryptocurrency developments"
        else:
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

        request_obj = QueryRequest(
            query=query,
            filter_date=query_filter_date,
            force_refresh=force_refresh,
            skip_embeddings=True,
            skip_batching=False,
            select_board=select_board,
            task_id=task_id,
            use_background=use_background,
            character_slug=None
        )

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

        result = await base_query(
            request=request_obj,
            background_tasks=background_tasks)

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

@router.post("/nl_query", response_model=NLQueryResponse)
async def natural_language_query(
    request: NLQueryRequest,
    agent: KnowledgeAgent = Depends(get_agent),
    save_result: bool = Query(True, description="Whether to save query results to disk")
) -> Dict[str, Any]:
    """
    Process a natural language query against the database using LLM-generated SQL.
    """
    start_time = time.time()

    from config.env_loader import detect_environment, is_replit_environment
    env_type = detect_environment()

    if not is_replit_environment():
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

        sql_generator = LLMSQLGenerator(agent)
        description = sql_generator.get_query_description(request.query)

        sql_query, params = await sql_generator.generate_sql(
            nl_query=request.query,
            provider=None,
            use_hybrid_approach=True
        )

        if request.limit and "LIMIT" not in sql_query.upper():
            sql_query += f" LIMIT %s"
            params.append(request.limit)

        from config.replit import PostgresDB
        db = PostgresDB()

        with db.get_connection() as conn:
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

        records = []
        for _, row in df.iterrows():
            record = {}
            for column, value in row.items():
                if column in ['author', 'channel_name', 'inserted_at']:
                    continue

                if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                    record[column] = value.isoformat()
                else:
                    record[column] = value

                if column == 'content' and isinstance(value, str) and len(value) > 500:
                    record[column] = value[:500] + "..."

            records.append(record)

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        metadata = {
            "processing_time_ms": execution_time_ms,
            "sql_generation_method": "llm",
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "providers_used": {
                "enhancer": sql_generator.PROVIDER_ENHANCER.value,
                "generator": sql_generator.PROVIDER_GENERATOR.value
            }
        }

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

        if save_result:
            try:
                task_id = generate_task_id(prefix="nlquery")
                
                from knowledge_agents.utils import save_query_output
                base_path = Path(Config.get_paths().get('generated_data', 'data/generated_data'))
                
                response_for_saving = {
                    "task_id": task_id,
                    "query": request.query,
                    "sql": sql_query,
                    "description": description,
                    "data": records,
                    "metadata": metadata.copy()
                }
                
                json_path, _ = await save_query_output(
                    response=response_for_saving,
                    base_path=base_path,
                    task_id=task_id,
                    query=request.query,
                    logger=logger,
                    include_embeddings=False
                )
                
                if json_path:
                    if "saved_files" not in response["metadata"]:
                        response["metadata"]["saved_files"] = {}
                    response["metadata"]["saved_files"]["json"] = str(json_path)
                    response["metadata"]["task_id"] = task_id
                    
                    from config.env_loader import is_replit_environment
                    response["metadata"]["storage_type"] = "object_storage" if is_replit_environment() else "filesystem"
                    
                    logger.info(f"Saved NL query results to {json_path}")
                
            except Exception as save_error:
                logger.warning(f"Could not save NL query results to disk: {save_error}")

        if request.format_for_llm:
            response = format_response_for_llm(response)

        log_endpoint_call(
            logger=logger, 
            endpoint="/nl_query",
            method="POST", 
            duration_ms=execution_time_ms,
            params={"query": request.query, "limit": request.limit, "record_count": len(records)}
        )

        return response

    except NLQueryParsingError as e:
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
        e.log_error(logger)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict()
        )
    except Exception as e:
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

# Note: Removed duplicate /api/v1/nl_query endpoint as per plan
# The /nl_query endpoint above will be accessible at /api/v1/nl_query through router prefix
