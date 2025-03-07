import os
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

from knowledge_agents.data_ops import DataConfig, DataOperations
from config.logging_config import get_logger
from config.env_loader import is_replit_environment, is_docker_environment

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
    data_ops = DataOperations(DataConfig.from_config())
    setup_marker = data_ops.config.root_data_path / ".setup_complete"
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
            
            # Initialize data operations with force_refresh=True for initial setup
            data_ops = DataOperations(DataConfig.from_config(force_refresh=True))
            data_dir = data_ops.config.root_data_path
            stratified_dir = data_ops.config.stratified_data_path
            
            # Create data directories
            data_dir.mkdir(parents=True, exist_ok=True)
            stratified_dir.mkdir(parents=True, exist_ok=True)
            
            # Define marker file paths
            init_marker = data_dir / '.initialization_in_progress'
            completion_marker = data_dir / '.initialization_complete'
            state_file = data_dir / '.initialization_state'
            embeddings_complete_marker = stratified_dir / '.embeddings_complete'
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
                # First phase: Data loading and stratification
                init_timeout = int(os.getenv("INIT_WAIT_TIME", "3600"))  # 60 minutes default
                try:
                    # First check if another worker has already completed embeddings
                    embeddings_path = stratified_dir / 'embeddings.npz'
                    thread_id_map_path = stratified_dir / 'thread_id_map.json'
                    
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
                            
                            # Generate embeddings - the method now handles creating the completion marker
                            await data_ops.generate_embeddings(force_refresh=True)
                    
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


# Import the standard app creation function
from .app import create_app

# Extend the app with additional startup/shutdown handlers if needed
def extend_app(app: FastAPI) -> FastAPI:
    """Add additional event handlers to the standard app."""
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize data processing during startup with robust cross-platform locking."""
        logger.info("Starting application initialization...")
        
        # Get worker identification
        worker_id = get_worker_id()
        logger.info(f"Worker {worker_id} starting up")
        
        # Log environment detection
        is_docker = is_docker_environment()
        logger.info(f"Docker environment detected: {is_docker}")
        
        # Create initialization marker files
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        init_marker = data_dir / ".initialization_in_progress"
        completion_marker = data_dir / ".initialization_complete"
        lock_file = data_dir / ".worker.lock"
        worker_specific_lock = data_dir / f".worker.lock.{worker_id}"
        
        # Explicitly set DOCKER_ENV if detected but not set
        if is_docker and os.environ.get("DOCKER_ENV", "").lower() != "true":
            logger.info("Setting DOCKER_ENV=true based on environment detection")
            os.environ["DOCKER_ENV"] = "true"
        
        try:
            # Determine if this worker should handle initialization
            should_initialize = is_primary_worker()
            
            if should_initialize:
                logger.info("Primary worker starting initialization")
                
                # Wait for setup to complete first if in Docker
                setup_timeout = int(os.getenv('SETUP_TIMEOUT', '120'))
                if os.getenv("DOCKER_ENV") == "true":
                    if not await wait_for_setup_complete(timeout=setup_timeout):
                        logger.warning("Setup not completed within timeout, proceeding anyway")
                
                # Create data directories if they don't exist
                data_config = DataConfig.from_config()
                os.makedirs(data_config.root_data_path, exist_ok=True)
                os.makedirs(data_config.stratified_data_path, exist_ok=True)
                
                # Log data status
                complete_data_path = data_config.root_data_path / 'complete_data.csv'
                embeddings_path = data_config.stratified_data_path / 'embeddings.npz'
                
                logger.info(f"Data file exists: {complete_data_path.exists()}")
                logger.info(f"Embeddings file exists: {embeddings_path.exists()}")
                
                # Log environment information
                logger.info(f"Environment: {os.environ.get('ENV_TYPE', 'unknown')}")
                logger.info(f"API workers: {os.environ.get('API_WORKERS', '1')}")
                logger.info(f"Data update interval: {os.environ.get('DATA_UPDATE_INTERVAL', '3600')} seconds")
                
                # Start background data initialization if needed
                if not init_marker.exists() and not complete_data_path.exists():
                    logger.info("Starting data initialization in background task")
                    from fastapi.background import BackgroundTasks
                    from .app import initialize_data_in_background
                    
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(initialize_data_in_background, data_config)
                    logger.info("Background data initialization task scheduled")
                else:
                    logger.info("Data initialization not needed or already in progress")
                
                # Clean up any stale marker files
                if completion_marker.exists():
                    try:
                        completion_marker.unlink()
                        logger.info(f"Cleaned up marker file: {completion_marker}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up marker file: {e}")
                
                # Schedule periodic cleanup of old results
                from .routes import _cleanup_old_results
                
                async def run_periodic_cleanup():
                    """Run the cleanup task periodically."""
                    cleanup_interval = int(os.getenv('RESULT_CLEANUP_INTERVAL', '300'))  # Default: 5 minutes
                    logger.info(f"Starting periodic result cleanup (interval: {cleanup_interval}s)")
                    while True:
                        try:
                            await asyncio.sleep(cleanup_interval)
                            await _cleanup_old_results()
                        except Exception as e:
                            logger.error(f"Error in periodic cleanup: {e}")
                            await asyncio.sleep(60)  # Wait a minute before retrying
                
                # Start the cleanup task
                asyncio.create_task(run_periodic_cleanup())
                logger.info("Periodic result cleanup scheduled")
                
                logger.info("Primary worker initialization complete")
            else:
                logger.info(f"Worker {worker_id} is not primary, waiting for initialization...")
                # Non-primary workers just wait for initialization to complete
                try:
                    await wait_for_initialization()
                    logger.info("Initialization complete, API ready")
                except Exception as e:
                    logger.error(f"Error waiting for initialization: {e}")
                    
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            logger.error(traceback.format_exc())
            
        logger.info("API startup complete")
    
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

# Function to get the appropriate app instance based on environment
def get_app() -> FastAPI:
    """Get the appropriate FastAPI application based on current environment."""
    # Create the base app
    app = create_app()
    
    # Apply environment-specific extensions
    if is_replit_environment():
        logger.info("Configuring app for Replit environment")
        from config.env_loader import configure_replit_environment
        configure_replit_environment()
        
        # Add Replit-specific routes if needed
        @app.get("/health_replit")
        async def health_replit():
            """Health check endpoint specifically for Replit."""
            from .routes import health_check_replit
            return await health_check_replit()
    elif is_docker_environment():
        logger.info("Configuring app for Docker environment")
        from config.env_loader import configure_docker_environment
        configure_docker_environment()
        
        # Apply standard extensions
        app = extend_app(app)
    else:
        # Apply standard extensions for local environment
        logger.info("Configuring app for local environment")
        app = extend_app(app)
        
    return app

# Create the application instance
app = get_app()

async def wait_for_initialization(timeout: int = 600) -> bool:
    """Wait for data initialization to complete.
    
    This function checks for the completion marker file that indicates
    the primary worker has finished initializing the data.
    
    Args:
        timeout: Maximum time to wait in seconds (default: 10 minutes)
        
    Returns:
        bool: True if initialization completed, False if timed out
    """
    data_dir = Path("data")
    completion_marker = data_dir / ".initialization_complete"
    init_marker = data_dir / ".initialization_in_progress"
    
    start_time = asyncio.get_event_loop().time()
    check_interval = int(os.getenv('INIT_CHECK_INTERVAL', '5'))
    logger.info(f"Waiting for initialization to complete (timeout: {timeout}s, check interval: {check_interval}s)...")
    
    while not completion_marker.exists():
        # Check if initialization is still in progress
        if not init_marker.exists():
            # If neither marker exists, initialization might have failed
            logger.warning("Initialization marker not found, initialization may have failed")
            # Check if data exists anyway
            data_config = DataConfig.from_config()
            complete_data_path = data_config.root_data_path / 'complete_data.csv'
            embeddings_path = data_config.stratified_data_path / 'embeddings.npz'
            
            if complete_data_path.exists() and embeddings_path.exists():
                logger.info("Data files exist despite missing markers, considering initialization complete")
                return True
        
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            logger.error(f"Initialization completion timeout after {elapsed:.1f}s")
            return False
            
        remaining = timeout - elapsed
        logger.info(f"Waiting for initialization to complete... ({remaining:.1f}s remaining)")
        await asyncio.sleep(check_interval)
    
    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"Initialization completed in {elapsed:.1f}s")
    return True