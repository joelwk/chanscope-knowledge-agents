import os
import logging
import asyncio
# Handle fcntl import for Windows compatibility
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # fcntl not available on Windows
    HAS_FCNTL = False
from typing import List
from functools import lru_cache
from pathlib import Path
import traceback
import multiprocessing
import json
from datetime import datetime, timedelta
import time
import pytz

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from knowledge_agents.data_ops import DataConfig, DataOperations
from .routes import router as api_router
from config.logging_config import get_logger

# Setup logging using centralized configuration
logger = get_logger(__name__)

# Add global initialization flags and locks
_data_processing_initialized = False
_initialization_lock = asyncio.Lock()
_initialization_error = None

def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    logger.info(f"Loaded environment: {get_environment()}")

@lru_cache()
def get_environment() -> str:
    """Get the current environment."""
    return os.getenv("ENVIRONMENT", "development")

@lru_cache()
def get_api_urls() -> List[str]:
    """Get list of allowed API URLs based on environment."""
    env = get_environment()
    if env == "production":
        return ["https://api.yourdomain.com"]
    elif env == "staging":
        return ["https://staging-api.yourdomain.com"]
    else:
        return ["http://localhost",
                "http://localhost:8000",
                "http://127.0.0.1",
                "http://127.0.0.1:8000"]

async def wait_for_setup_complete(timeout: int = None) -> bool:
    """Wait for setup.sh to complete initialization."""
    setup_marker = Path("data/.setup_complete")
    start_time = asyncio.get_event_loop().time()
    # Use environment variable for timeout if available
    timeout = timeout or int(os.getenv('SETUP_TIMEOUT', '120'))
    check_interval = int(os.getenv('SETUP_CHECK_INTERVAL', '5'))
    logger.info(f"Waiting for setup to complete (timeout: {timeout}s, check interval: {check_interval}s)...")
    while not setup_marker.exists():
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            logger.error(f"Setup completion timeout after {elapsed:.1f}s")
            return False
        remaining = timeout - elapsed
        logger.info(f"Waiting for setup to complete... ({remaining:.1f}s remaining)")
        await asyncio.sleep(check_interval)
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"Setup completed in {elapsed:.1f}s")
    return True

async def initialize_data_processing() -> None:
    """Initialize data processing with improved worker coordination."""
    global _data_processing_initialized, _initialization_lock, _initialization_error
    if _data_processing_initialized:
        return
    try:
        async with _initialization_lock:
            if _data_processing_initialized:
                return
            logger.info("Starting data processing initialization")
            
            # Define marker file paths
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            init_marker = data_dir / '.initialization_in_progress'
            completion_marker = data_dir / '.initialization_complete'
            state_file = data_dir / '.initialization_state'
            embeddings_complete_marker = data_dir / 'stratified/.embeddings_complete'
            lock_file = data_dir / '.worker.lock'
            
            # Clean up any stale marker files at start
            for marker in [init_marker, completion_marker, state_file, lock_file]:
                try:
                    if marker.exists():
                        marker.unlink()
                        logger.info(f"Cleaned up stale marker: {marker}")
                except FileNotFoundError:
                    # File might have been removed by another process, that's okay
                    logger.info(f"Marker already removed: {marker}")
                except Exception as e:
                    logger.warning(f"Error cleaning up marker {marker}: {e}")
            
            # Create fresh initialization marker
            init_marker.touch()
            
            try:
                # Initialize data operations with force_refresh=True for initial setup
                data_ops = DataOperations(DataConfig.from_config(force_refresh=True))
                
                # First phase: Data loading and stratification
                init_timeout = int(os.getenv("INIT_WAIT_TIME", "3600"))  # 60 minutes default
                try:
                    # First check if another worker has already completed embeddings
                    embeddings_path = data_dir / 'stratified/embeddings.npz'
                    thread_id_map_path = data_dir / 'stratified/thread_id_map.json'
                    
                    if embeddings_complete_marker.exists() and embeddings_path.exists() and thread_id_map_path.exists():
                        logger.info("Embeddings already generated by another worker, skipping generation")
                        # We'll still load data but skip embeddings generation
                        async with asyncio.timeout(300):  # 5 minutes should be enough for loading
                            await data_ops.ensure_data_ready(force_refresh=False)
                    else:
                        # Do full initialization including embeddings with longer timeout
                        async with asyncio.timeout(init_timeout):
                            # Ensure data is ready (this will load and stratify the data)
                            await data_ops.ensure_data_ready(force_refresh=True)
                            
                            # Create embeddings completion marker
                            embeddings_complete_marker.touch()
                    
                    # Write completion state
                    state = {
                        'timestamp': datetime.now().isoformat(),
                        'worker_id': get_worker_id(),
                        'status': 'complete'
                    }
                    with open(state_file, 'w') as sf:
                        json.dump(state, sf)
                        sf.flush()
                        os.fsync(sf.fileno())
                    
                    # Create completion marker last
                    completion_marker.touch()
                    
                    # Remove initialization marker
                    if init_marker.exists():
                        init_marker.unlink()
                    
                    # Mark initialization as complete
                    _data_processing_initialized = True
                    logger.info("Data processing initialization completed successfully")
                except asyncio.TimeoutError:
                    logger.error(f"Initialization timed out after {init_timeout} seconds")
                    # Check if embeddings were completed during our timeout
                    if embeddings_path.exists() and thread_id_map_path.exists():
                        logger.info("Embeddings exist despite timeout, continuing...")
                        _data_processing_initialized = True
                    else:
                        # Clean up markers on timeout
                        for marker in [init_marker, completion_marker]:
                            try:
                                if marker.exists():
                                    marker.unlink()
                                    logger.info(f"Cleaned up marker on timeout: {marker}")
                            except FileNotFoundError:
                                # File might have been removed by another process, that's okay
                                logger.info(f"Marker already removed on timeout: {marker}")
                            except Exception as e:
                                logger.warning(f"Error cleaning up marker {marker} on timeout: {e}")
                        raise
                    
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
                logger.error(traceback.format_exc())
                # Clean up markers on failure
                for marker in [init_marker, completion_marker]:
                    try:
                        if marker.exists():
                            marker.unlink()
                            logger.info(f"Cleaned up marker on error: {marker}")
                    except FileNotFoundError:
                        # File might have been removed by another process, that's okay
                        logger.info(f"Marker already removed on error: {marker}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up marker {marker} on error: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Error initializing data processing: {e}")
        _initialization_error = str(e)
        raise

def get_worker_id() -> str:
    """Get a unique worker ID with improved process identification.
    
    This function uses multiple strategies to identify the worker:
    1. Check for an explicit WORKER_ID environment variable
    2. Use the current process info and Uvicorn worker number
    3. Fallback to PID + hostname for uniqueness
    
    Returns:
        str: A unique worker identifier
    """
    # First check environment variable
    worker_id = os.getenv('WORKER_ID')
    if worker_id:
        return worker_id
        
    # Get hostname for uniqueness across machines
    try:
        import socket
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown-host"
        
    # Get current process info
    current_process = multiprocessing.current_process()
    
    # Check if we're the main process
    if current_process.name == 'MainProcess':
        return f"Main-{hostname}-{os.getpid()}"
    
    # For Uvicorn worker processes
    if hasattr(current_process, '_identity') and current_process._identity:
        worker_num = current_process._identity[0]
        return f"Worker-{hostname}-{worker_num}-{os.getpid()}"
    
    # For Gunicorn workers, check specific environment variables
    if os.getenv('GUNICORN_WORKER_ID'):
        return f"Gunicorn-{hostname}-{os.getenv('GUNICORN_WORKER_ID')}"
    
    # If we can't determine the worker ID, use PID + timestamp for uniqueness
    import time
    timestamp = int(time.time())
    return f"Worker-{hostname}-{os.getpid()}-{timestamp}"

def is_primary_worker() -> bool:
    """Check if current process is the primary worker with improved detection.
    
    This function identifies the primary worker that should perform initialization.
    It uses multiple strategies to ensure only one worker is designated as primary.
    
    Returns:
        bool: True if this worker should be considered the primary worker
    """
    # Get current process info
    current_process = multiprocessing.current_process()
    worker_id = get_worker_id()
    
    # Check for explicit environment variable flag
    primary_flag = os.getenv('PRIMARY_WORKER', '').lower()
    if primary_flag in ('1', 'true', 'yes'):
        logger.info(f"Worker {worker_id} designated as primary by environment variable")
        return True
    elif primary_flag in ('0', 'false', 'no'):
        logger.info(f"Worker {worker_id} explicitly set as non-primary by environment variable")
        return False
    
    # Consider the following cases as primary:
    # 1. Process is MainProcess (single worker mode)
    # 2. We're running as the first worker in a multi-worker setup
    is_main_process = current_process.name == 'MainProcess'
    
    # Check for Uvicorn worker ID
    is_first_worker = False
    if hasattr(current_process, '_identity') and current_process._identity:
        is_first_worker = current_process._identity[0] == 1
    
    # Primary determination logic
    is_primary = is_main_process or is_first_worker
    
    # Use file-based primary election as a fallback
    if not is_primary:
        try:
            primary_marker = Path("data/.primary_worker")
            primary_marker.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to create the primary marker file if it doesn't exist
            if not primary_marker.exists():
                try:
                    # Use exclusive creation to ensure only one worker succeeds
                    with open(primary_marker, 'x') as f:
                        f.write(worker_id)
                    logger.info(f"Worker {worker_id} elected as primary through file marker")
                    is_primary = True
                except FileExistsError:
                    # Another worker beat us to it
                    logger.info(f"Primary worker already elected, {worker_id} is secondary")
            else:
                # Check if the primary marker is stale (older than 10 minutes)
                try:
                    marker_age = time.time() - primary_marker.stat().st_mtime
                    if marker_age > 600:  # 10 minutes in seconds
                        # Try to take over as primary
                        with open(primary_marker, 'w') as f:
                            f.write(worker_id)
                        logger.info(f"Worker {worker_id} taking over as primary (stale marker: {marker_age:.1f}s)")
                        is_primary = True
                    else:
                        # Check if we're already the primary worker
                        try:
                            with open(primary_marker, 'r') as f:
                                current_primary = f.read().strip()
                            if current_primary == worker_id:
                                logger.info(f"Worker {worker_id} confirmed as existing primary")
                                is_primary = True
                        except Exception as e:
                            logger.warning(f"Error reading primary marker: {e}")
                except Exception as e:
                    logger.warning(f"Error checking primary marker age: {e}")
        except Exception as e:
            logger.warning(f"Error in primary worker election: {e}")
    
    logger.info(f"Worker {worker_id} primary status: {is_primary}")
    return is_primary

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Load environment variables
    load_environment()
    
    # Create FastAPI app with enhanced metadata
    app = FastAPI(
        title="Knowledge Agent API",
        description="API for processing and analyzing text using AI models",
        version="1.0.0",
        docs_url="/docs" if get_environment() != "production" else None,
        redoc_url="/redoc" if get_environment() != "production" else None
    )
    
    # Add middleware from app.py
    from .app import add_middleware
    add_middleware(app)
    
    # Include API routes with proper prefix
    app.include_router(api_router, prefix="/api/v1", tags=["knowledge_agents"])

    @app.on_event("startup")
    async def startup_event():
        """Initialize data processing during startup with robust cross-platform locking."""
        logger.info("Starting application initialization...")
        
        # Get worker identification
        worker_id = get_worker_id()
        logger.info(f"Worker {worker_id} starting up")
        
        # Create initialization marker files
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        init_marker = data_dir / ".initialization_in_progress"
        completion_marker = data_dir / ".initialization_complete"
        lock_file = data_dir / ".worker.lock"
        worker_specific_lock = data_dir / f".worker.lock.{worker_id}"
        
        try:
            # Determine if this worker should handle initialization
            should_initialize = is_primary_worker()
            
            if should_initialize:
                logger.info("Primary worker starting initialization")
                
                # Wait for setup to complete first if in Docker
                setup_timeout = int(os.getenv('SETUP_TIMEOUT', '120'))
                if os.getenv("DOCKER_ENV") == "true":
                    if not await wait_for_setup_complete(timeout=setup_timeout):
                        logger.error(f"Setup not completed after {setup_timeout}s")
                        return
                    logger.info("Setup completed successfully")
                
                # Clean up any existing markers at startup
                for marker in [init_marker, completion_marker, worker_specific_lock]:
                    if marker.exists():
                        try:
                            marker.unlink()
                            logger.info(f"Cleaned up marker file: {marker}")
                        except Exception as e:
                            logger.warning(f"Could not clean up marker {marker}: {e}")
                
                # Create parent directory for lock files if needed
                lock_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Cross-platform locking strategy
                lock_acquired = False
                try:
                    if HAS_FCNTL:
                        # Linux-style file locking
                        with open(lock_file, 'w') as f:
                            f.write(f"{worker_id}\n")
                            f.flush()
                            try:
                                # Try non-blocking exclusive lock
                                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                lock_acquired = True
                                logger.info(f"Worker {worker_id} acquired fcntl lock")
                            except (IOError, OSError) as e:
                                logger.info(f"Another worker already has the lock: {e}")
                    else:
                        # Windows-style lock using exclusive file creation
                        try:
                            # Create worker-specific lock file exclusively
                            with open(worker_specific_lock, 'x') as f:
                                f.write(f"{worker_id}\n{datetime.now(pytz.UTC).isoformat()}")
                            
                            # Try to acquire global lock - write our ID to it
                            with open(lock_file, 'w') as f:
                                f.write(f"{worker_id}\n{datetime.now(pytz.UTC).isoformat()}")
                                f.flush()
                                
                            # Check if any other worker lock files exist
                            other_locks = [
                                p for p in data_dir.glob(".worker.lock.*") 
                                if p != worker_specific_lock and p.is_file()
                            ]
                            
                            # Check if other locks are stale (> 5 minutes old)
                            active_locks = []
                            for other_lock in other_locks:
                                try:
                                    lock_age = time.time() - other_lock.stat().st_mtime
                                    if lock_age < 300:  # 5 minutes
                                        active_locks.append(other_lock)
                                    else:
                                        logger.info(f"Found stale lock file: {other_lock} (age: {lock_age:.1f}s)")
                                        try:
                                            other_lock.unlink()
                                        except Exception as e:
                                            logger.warning(f"Could not remove stale lock: {e}")
                                except Exception as e:
                                    logger.warning(f"Error checking lock file {other_lock}: {e}")
                                    
                            if not active_locks:
                                lock_acquired = True
                                logger.info(f"Worker {worker_id} acquired Windows-style lock")
                            else:
                                logger.info(f"Other active workers found: {[l.name for l in active_locks]}")
                        except FileExistsError:
                            logger.info(f"Worker-specific lock file already exists for {worker_id}")
                    
                    if lock_acquired:
                        # Create initialization marker to indicate we're starting
                        init_marker.touch()
                        
                        # Perform actual initialization
                        await initialize_data_processing()
                        
                        # Mark initialization as complete
                        completion_marker.touch()
                        logger.info("Primary worker completed initialization")
                    else:
                        # Wait for initialization to complete
                        logger.info(f"Worker {worker_id} waiting for primary worker initialization")
                        await wait_for_initialization()
                    
                except Exception as e:
                    logger.error(f"Primary worker initialization failed: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Clean up markers on failure
                    for marker in [init_marker, completion_marker]:
                        if marker.exists():
                            try:
                                marker.unlink()
                            except Exception as cleanup_error:
                                logger.warning(f"Error cleaning up marker {marker}: {cleanup_error}")
                    raise
                finally:
                    # Release locks
                    if HAS_FCNTL and lock_acquired:
                        try:
                            with open(lock_file, 'r+') as f:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                                logger.info(f"Worker {worker_id} released fcntl lock")
                        except Exception as e:
                            logger.warning(f"Error releasing fcntl lock: {e}")
                    
                    # Remove worker-specific lock if used
                    if worker_specific_lock.exists():
                        try:
                            worker_specific_lock.unlink()
                            logger.info(f"Removed worker-specific lock: {worker_specific_lock}")
                        except Exception as e:
                            logger.warning(f"Error removing worker-specific lock: {e}")
            else:
                logger.info(f"Secondary worker {worker_id} waiting for primary worker initialization")
                await wait_for_initialization()
                
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            logger.error(traceback.format_exc())
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on application shutdown with improved error handling.
        
        This function ensures:
        1. All locks and marker files are properly released
        2. Temporary resources are cleaned up
        3. The shutdown is logged for troubleshooting
        """
        worker_id = get_worker_id()
        logger.info(f"Worker {worker_id} shutting down, cleaning up resources...")
        
        # Define all marker files that need cleanup
        data_dir = Path("data")
        marker_files = [
            '.initialization_complete',
            '.initialization_in_progress',
            '.initialization_state',
            '.worker.lock',
            f'.worker.lock.{worker_id}',
            '.embedding_task_in_progress',
            f'.worker_{worker_id}_in_progress',
            f'.embedding_task_{worker_id}'
        ]
        
        # Additional worker-specific files to clean up
        worker_files = list(data_dir.glob(f"*{worker_id}*"))
        
        # Clean up marker files
        cleanup_errors = []
        for marker in marker_files:
            marker_path = data_dir / marker
            if marker_path.exists():
                try:
                    marker_path.unlink()
                    logger.info(f"Cleaned up marker file: {marker}")
                except Exception as e:
                    error_msg = f"Error cleaning up {marker}: {e}"
                    logger.error(error_msg)
                    cleanup_errors.append(error_msg)
        
        # Clean up any additional worker-specific files
        for file_path in worker_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up worker file: {file_path}")
                except Exception as e:
                    error_msg = f"Error cleaning up worker file {file_path}: {e}"
                    logger.error(error_msg)
                    cleanup_errors.append(error_msg)
        
        # Check for stale locks from other workers and attempt to release them
        # Only the primary worker should do this to avoid race conditions
        if is_primary_worker():
            try:
                # Find stale worker locks (older than 10 minutes)
                stale_locks = []
                for lock_file in data_dir.glob(".worker.lock.*"):
                    try:
                        lock_age = time.time() - lock_file.stat().st_mtime
                        if lock_age > 600:  # 10 minutes in seconds
                            stale_locks.append(lock_file)
                    except Exception as e:
                        logger.warning(f"Error checking lock file age {lock_file}: {e}")
                
                # Remove stale locks
                for lock_file in stale_locks:
                    try:
                        lock_file.unlink()
                        logger.info(f"Cleaned up stale lock: {lock_file}")
                    except Exception as e:
                        logger.warning(f"Error removing stale lock {lock_file}: {e}")
            except Exception as e:
                logger.error(f"Error cleaning up stale locks: {e}")
        
        # Release any active FCNTL locks
        if HAS_FCNTL:
            lock_file = data_dir / ".worker.lock"
            if lock_file.exists():
                try:
                    with open(lock_file, 'r+') as f:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            logger.info("Released fcntl lock on shutdown")
                        except Exception as e:
                            logger.warning(f"Error releasing fcntl lock: {e}")
                except Exception as e:
                    logger.warning(f"Error opening lock file: {e}")
        
        # Log cleanup results
        if cleanup_errors:
            logger.warning(f"Shutdown completed with {len(cleanup_errors)} cleanup errors")
        else:
            logger.info("Shutdown completed successfully with clean resource release")
        
        # Add a marker indicating clean shutdown for troubleshooting
        try:
            shutdown_marker = data_dir / f".shutdown_marker_{worker_id}"
            with open(shutdown_marker, 'w') as f:
                f.write(f"{worker_id} shutdown at {datetime.now(pytz.UTC).isoformat()}")
        except Exception as e:
            logger.warning(f"Error creating shutdown marker: {e}")
    
    # Add health check endpoint
    @app.get("/api/v1/health")
    async def health_check():
        """API health check endpoint."""
        from datetime import datetime, timezone
        global _initialization_error
        status = "healthy"
        details = {}
        
        # Check initialization status
        if _initialization_error:
            status = "degraded"
            details["initialization_error"] = _initialization_error
        # Check marker files
        init_marker = Path("data/.initialization_in_progress")
        completion_marker = Path("data/.initialization_complete")
        details["initialization_status"] = "complete" if completion_marker.exists() else "in_progress" if init_marker.exists() else "not_started"
        return {
            
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": os.getenv("FASTAPI_ENV", "development"),
            "version": "1.0.0",
            "details": details
        }
    
    return app

async def wait_for_initialization(timeout: int = 900) -> None:
    """Wait for initialization to complete with improved state checking and stale lock detection.
    
    This function implements a robust waiting mechanism that can:
    1. Detect when initialization is complete
    2. Identify and recover from stale initialization processes
    3. Handle timeouts gracefully
    
    Args:
        timeout: Maximum time to wait in seconds (default: 15 minutes)
    """
    data_dir = Path("data")
    completion_marker = data_dir / '.initialization_complete'
    init_marker = data_dir / '.initialization_in_progress'
    state_file = data_dir / '.initialization_state'
    lock_file = data_dir / '.worker.lock'
    
    start_time = asyncio.get_event_loop().time()
    check_interval = 5  # seconds
    stale_threshold = 600  # 10 minutes in seconds
    
    logger.info(f"Waiting for initialization (timeout: {timeout}s)...")
    
    while not completion_marker.exists() or not is_initialization_fresh(state_file):
        # Check for timeout
        time_elapsed = asyncio.get_event_loop().time() - start_time
        if time_elapsed > timeout:
            logger.warning(f"Timeout after {time_elapsed:.1f}s waiting for initialization")
            
            # Check if initialization is actually in progress
            if init_marker.exists():
                # Check if the marker is stale
                marker_age = time.time() - init_marker.stat().st_mtime
                if marker_age > stale_threshold:
                    logger.warning(f"Found stale initialization marker (age: {marker_age:.1f}s), attempting recovery")
                    try:
                        # Try to clean up stale markers
                        try:
                            init_marker.unlink()
                            logger.info(f"Removed stale initialization marker")
                        except FileNotFoundError:
                            logger.info(f"Stale initialization marker already removed")
                        except Exception as e:
                            logger.warning(f"Error removing stale initialization marker: {e}")
                            
                        if lock_file.exists():
                            try:
                                lock_file.unlink()
                                logger.info(f"Removed stale lock file")
                            except FileNotFoundError:
                                logger.info(f"Stale lock file already removed")
                            except Exception as e:
                                logger.warning(f"Error removing stale lock file: {e}")
                        
                        # Create our own marker and assume control
                        init_marker.touch()
                        logger.info("Taking over initialization after stale process")
                        
                        # Return to allow the caller to retry initialization
                        return
                    except Exception as e:
                        logger.error(f"Failed to recover from stale initialization: {e}")
                else:
                    logger.info(f"Initialization is still in progress (marker age: {marker_age:.1f}s)")
            else:
                logger.warning("No initialization markers found, initialization may have failed")
            break
            
        # Check if state file exists but is stale
        if state_file.exists():
            try:
                state_age = time.time() - state_file.stat().st_mtime
                if state_age > stale_threshold and init_marker.exists():
                    logger.warning(f"Found stale state file (age: {state_age:.1f}s), attempting recovery")
                    try:
                        # Try to clean up stale markers
                        try:
                            init_marker.unlink()
                            logger.info(f"Removed stale initialization marker")
                        except FileNotFoundError:
                            logger.info(f"Stale initialization marker already removed")
                        except Exception as e:
                            logger.warning(f"Error removing stale initialization marker: {e}")
                            
                        if lock_file.exists():
                            try:
                                lock_file.unlink()
                                logger.info(f"Removed stale lock file")
                            except FileNotFoundError:
                                logger.info(f"Stale lock file already removed")
                            except Exception as e:
                                logger.warning(f"Error removing stale lock file: {e}")
                        return
                    except Exception as e:
                        logger.error(f"Failed to recover from stale state: {e}")
            except Exception as e:
                logger.warning(f"Error checking state file age: {e}")
        
        # Log progress periodically
        if int(time_elapsed) % 30 == 0:  # Log every 30 seconds
            time_remaining = timeout - time_elapsed
            logger.info(f"Still waiting for initialization... ({time_remaining:.1f}s remaining)")
            
            # Check for signs of progress
            if init_marker.exists():
                try:
                    marker_age = time.time() - init_marker.stat().st_mtime
                    logger.info(f"Initialization marker age: {marker_age:.1f}s")
                except Exception:
                    pass
        
        await asyncio.sleep(check_interval)
    
    if completion_marker.exists() and is_initialization_fresh(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Initialization completed by worker {state.get('worker_id')} at {state.get('timestamp')}")
        except Exception as e:
            logger.warning(f"Error reading initialization state: {e}")
    else:
        logger.warning("Proceeding without complete initialization")

def is_initialization_fresh(state_file: Path, max_age_minutes: int = 30) -> bool:
    """Check if initialization state is fresh enough with improved error handling.
    
    Args:
        state_file: Path to the state file
        max_age_minutes: Maximum age in minutes for the state to be considered fresh
        
    Returns:
        bool: True if state is fresh, False otherwise
    """
    if not state_file.exists():
        return False
    try:
        # Check file size to avoid trying to parse empty or corrupted files
        if state_file.stat().st_size == 0:
            logger.warning("State file exists but is empty")
            return False
            
        with open(state_file, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.warning("State file exists but contains no data")
                return False
                
            # Try to parse the JSON
            state = json.loads(content)
            
        # Validate timestamp field exists
        if 'timestamp' not in state:
            logger.warning("State file missing timestamp field")
            return False
            
        # Parse timestamp with error handling
        try:
            last_init = datetime.fromisoformat(state.get('timestamp', ''))
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid timestamp format in state file: {e}")
            return False
            
        # Check freshness
        age = datetime.now() - last_init
        is_fresh = age < timedelta(minutes=max_age_minutes)
        
        if not is_fresh:
            logger.warning(f"State is stale: {age.total_seconds()/60:.1f} minutes old (max: {max_age_minutes} minutes)")
        
        return is_fresh
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted state file (JSON error): {e}")
        return False
    except Exception as e:
        logger.warning(f"Error checking initialization freshness: {e}")
        return False

def create_replit_app() -> FastAPI:
    """Create and configure the FastAPI application specifically for Replit environment.
    
    This function creates an app with Replit-specific configurations while maintaining
    compatibility with Docker and other environments.
    """
    # Load environment variables
    load_environment()
    
    # Set default port for Replit
    port = os.getenv("PORT", "80")
    api_port = os.getenv("API_PORT", port)
    host = os.getenv("HOST", "0.0.0.0")
    
    # Log port configuration and FastAPI version
    import fastapi
    logger.info(f"Configuring Replit app with HOST={host}, PORT={port}, API_PORT={api_port}")
    logger.info(f"Using FastAPI version: {fastapi.__version__}")
    logger.info(f"Replit environment: REPL_ID={os.getenv('REPL_ID', 'unknown')}, REPL_SLUG={os.getenv('REPL_SLUG', 'unknown')}")
    
    # Create FastAPI app with enhanced metadata
    app = FastAPI(
        title="Knowledge Agent API (Replit)",
        description="API for processing and analyzing text using AI models",
        version="1.0.0",
        docs_url="/docs",  # Always enable docs for Replit
        redoc_url="/redoc"  # Always enable redoc for Replit
    )
    
    # Add middleware from app.py
    from .app import add_middleware
    add_middleware(app)
    
    # Include the main API routes with the /api/v1 prefix
    app.include_router(api_router, prefix="/api/v1", tags=["knowledge_agents"])
    
    # Create a separate health check endpoint for the /api prefix
    # This avoids duplicate operation IDs while still providing basic functionality
    @app.get("/api/health")
    async def api_health_endpoint():
        """Health check endpoint at the API root level."""
        return {"status": "ok", "api": "healthy"}
    
    # Add a root path redirect
    @app.get("/")
    async def root_redirect():
        """Redirect root to docs in Replit environment."""
        return RedirectResponse(url="/docs")
    
    # Add a simple health check endpoint at the root level for Replit's health check system
    @app.get("/healthz")
    async def healthz():
        """Simple health check endpoint for Replit's health check system."""
        return {"status": "ok"}
    
    # Add specialized Replit health check at root level
    @app.get("/health_replit")
    async def root_health_replit():
        """API health check endpoint for Replit."""
        from datetime import datetime, timezone
        
        # Call the main health check logic
        from .routes import health_check_replit as routes_health_check
        health_data = await routes_health_check()
        
        # Add Replit port information
        health_data["port_config"] = {
            "api_port": api_port,
            "port": port,
            "host": host
        }
        
        return health_data
        
    @app.on_event("startup")
    async def startup_event():
        """Initialize data processing during startup."""
        logger.info("Starting application initialization in Replit environment...")
        worker_id = get_worker_id()
        logger.info(f"Replit worker {worker_id} starting up")
        
        # Create initialization marker files
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use more lightweight initialization for Replit
        should_initialize = True  # Always initialize in Replit environment
        
        if should_initialize:
            try:
                logger.info(f"Worker {worker_id} initializing data processing...")
                await initialize_data_processing()
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
                global _initialization_error
                _initialization_error = str(e)
                raise
        else:
            logger.info(f"Worker {worker_id} waiting for initialization...")
            try:
                logger.info("Waiting for initialization to complete...")
                await wait_for_initialization()
                logger.info("Initialization complete, API ready")
            except Exception as e:
                logger.error(f"Error waiting for initialization: {e}")
                raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        logger.info("Shutting down Replit API instance...")
        worker_id = get_worker_id()
        
        try:
            shutdown_marker = Path(f"data/.worker_shutdown_{worker_id}")
            with open(shutdown_marker, "w") as f:
                f.write(f"{worker_id} shutdown at {datetime.now(pytz.UTC).isoformat()}")
        except Exception as e:
            logger.warning(f"Error creating shutdown marker: {e}")
    
    return app

# Create application instance based on environment rather than directly using create_app()
from .app import get_app
app = get_app()
