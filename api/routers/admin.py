"""Admin and debug endpoints."""
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.routers.shared import get_batch_results
from config.settings import Config
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["admin"])

@router.post("/admin/cleanup")
async def trigger_cleanup(
    force: bool = Query(False, description="Force cleanup of all files regardless of age")
) -> Dict[str, Any]:
    """Admin endpoint to manually trigger cleanup of memory and disk storage."""
    start_time = time.time()
    memory_items_removed = 0
    disk_files_removed = 0
    
    try:
        _batch_results = get_batch_results()
        
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
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "memory_items_removed": memory_items_removed,
            "disk_files_removed": disk_files_removed,
            "duration_ms": round((time.time() - start_time) * 1000, 2)
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

@router.get("/metrics", include_in_schema=False)
async def metrics():
    """Metrics endpoint exposing cache stats and basic process metrics."""
    import psutil
    import os
    
    try:
        from api.cache import CACHE_HITS, CACHE_MISSES, CACHE_ERRORS
        
        # Get cache metrics
        if hasattr(CACHE_HITS, 'get_value'):
            hits = CACHE_HITS.get_value()
            misses = CACHE_MISSES.get_value()
            errors = CACHE_ERRORS.get_value()
        else:
            hits = CACHE_HITS
            misses = CACHE_MISSES
            errors = CACHE_ERRORS
        
        total_requests = hits + misses
        hit_ratio = (hits / total_requests * 100) if total_requests > 0 else 0
        
        # Get process stats
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "cache": {
                "hits": int(hits),
                "misses": int(misses),
                "errors": int(errors),
                "total_requests": int(total_requests),
                "hit_ratio": f"{hit_ratio:.2f}%"
            },
            "process": {
                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "uptime_seconds": round(time.time() - process.create_time(), 2)
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {
            "error": str(e),
            "cache": {"error": "unavailable"},
            "process": {"error": "unavailable"}
        }

