"""
Main API module for the Chanscope application.

This module provides the FastAPI application for the Chanscope approach,
supporting both file-based storage (Docker) and database storage (Replit).
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
# Import configuration utilities
from config.env_loader import load_environment, detect_environment, is_replit_environment
from config.logging_config import setup_logging
from config.settings import Config

# Import the unified data manager
from knowledge_agents.data_processing.chanscope_manager import ChanScopeDataManager
from config.chanscope_config import ChanScopeConfig

# Import API router from routes.py
from .routes import router as api_router

# Set up logging
setup_logging()
logger = logging.getLogger("api")

# Initialize environment variables
load_environment()

# Lifespan context manager for proper startup/shutdown handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to control application startup and shutdown.
    This ensures we can respond to health checks immediately while still
    preparing our data in the background.
    """
    # Clean logs on startup
    try:
        from knowledge_agents.utils import clean_logs
        await clean_logs()
        logger.info("Logs cleaned on startup")
    except Exception as e:
        logger.warning(f"Failed to clean logs on startup: {e}")
    
    # Set a flag to indicate the API is ready for basic health checks
    app.state.ready_for_health_checks = True
    logger.info("API ready for health checks")
    
    # Yield control back to FastAPI to start server
    # This allows the server to bind and respond to health checks immediately
    yield
    
    # ---- Code below runs AFTER the server has started ----
    logger.info("Server started, initiating background data processing if enabled.")
    # Initialize data in the background - don't block API startup
    try:
        # Only initialize if specified in environment
        auto_check_data = os.environ.get('AUTO_CHECK_DATA', 'true').lower() in ('true', '1', 'yes')
        logger.info(f"AUTO_CHECK_DATA environment variable: {os.environ.get('AUTO_CHECK_DATA', 'not set')}")
        logger.info(f"AUTO_CHECK_DATA parsed value: {auto_check_data}")
        
        if auto_check_data:
            logger.info("AUTO_CHECK_DATA is enabled, initiating data preparation in background")
            # Create the task but don't await it - let it run fully in background
            task = asyncio.create_task(initialize_background_data())
            # Add a name to the task for easier debugging
            task.set_name("background_data_init")
            logger.info(f"Created background task: {task.get_name()}")
        else:
            logger.info("AUTO_CHECK_DATA is disabled, skipping initial data preparation")
    except Exception as e:
        logger.error(f"Error during background data initialization startup: {e}", exc_info=True)
    
    # ---- Cleanup on shutdown ----
    logger.info("Shutting down Chanscope API...")
    # You might add specific cleanup logic here if needed before the application exits.

# Health check middleware to ensure the root endpoint always responds
class HealthCheckMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        # Very lightweight health check at the root path
        # This ensures that even if other parts of the app fail,
        # the health check endpoint will still respond
        if request.url.path == "/":
            return JSONResponse(content={"status": "ok"})
        return await call_next(request)

# Create FastAPI application with lifespan context manager
app = FastAPI(
    title="Chanscope API",
    description="API for the Chanscope data processing and query system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    # Disable OpenAPI docs in production for faster startup
    openapi_url="/openapi.json" if os.environ.get("FASTAPI_DEBUG", "false").lower() == "true" else None
)

# Add health check middleware first to ensure it intercepts root requests
# and always returns a health check response
app.add_middleware(HealthCheckMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create configuration and data manager
chanscope_config = ChanScopeConfig.from_env()
logger.info(f"Initialized ChanScopeConfig: {chanscope_config}")

# Create data manager using factory method for the appropriate environment
environment = detect_environment()
logger.info(f"Detected environment: {environment}")
data_manager = ChanScopeDataManager.create_for_environment(chanscope_config)
logger.info(f"Initialized ChanScopeDataManager for {environment} environment")

# Include API router with proper prefix
app.include_router(api_router, prefix="/api/v1", tags=["knowledge_agents"])

# Define request models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: int = 5
    force_refresh: bool = False
    skip_embeddings: bool = False

class QueryResponse(BaseModel):
    """Query response model."""
    status: str = "completed"
    query: str
    top_k: int
    chunks: list = []
    summary: str = ""
    metadata: Dict[str, Any]

# Dependency for data readiness
async def ensure_data_ready(
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> bool:
    """
    Ensure data is ready for use.
    
    Args:
        force_refresh: Whether to force refresh all data
        skip_embeddings: Whether to skip embedding generation
        
    Returns:
        True if data is ready
        
    Raises:
        HTTPException: If data is not ready
    """
    # Check if data is ready
    data_ready = await data_manager.is_data_ready(skip_embeddings=skip_embeddings)
    
    if not data_ready:
        # Start data preparation in the background
        await data_manager.ensure_data_ready(
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
        
        # Check if data is ready now
        data_ready = await data_manager.is_data_ready(skip_embeddings=skip_embeddings)
        
        if not data_ready:
            raise HTTPException(
                status_code=503,
                detail="Data is being prepared. Please try again later."
            )
    
    return True

# Background data initialization function used by __init__.py
async def initialize_data_in_background(data_config):
    """
    Initialize data processing in the background.
    
    This function is called by __init__.py's startup handler to prepare
    data asynchronously without blocking the API startup.
    
    Args:
        data_config: Configuration for data initialization
    """
    logger.info("Starting background data initialization...")
    try:
        # Use the existing data manager to initialize data
        force_refresh = os.environ.get('FORCE_DATA_REFRESH', 'false').lower() in ('true', '1', 'yes')
        skip_embeddings = os.environ.get('SKIP_EMBEDDINGS', 'false').lower() in ('true', '1', 'yes')
        environment = detect_environment()
        
        # Mark the operation in progress
        await data_manager.state_manager.mark_operation_start("background_initialization")
        
        # Log environment and configuration
        logger.info(f"Environment: {environment}")
        logger.info(f"Force refresh: {force_refresh}")
        logger.info(f"Skip embeddings: {skip_embeddings}")
        
        # If in Replit, verify PostgreSQL schema first
        if environment == 'replit':
            logger.info("Ensuring PostgreSQL schema is initialized for Replit environment")
            from config.replit import PostgresDB
            try:
                db = PostgresDB()
                db.initialize_schema()
                logger.info("PostgreSQL schema is ready")
            except Exception as e:
                logger.error(f"Error initializing PostgreSQL schema: {e}")
                # Still continue with data initialization
        
        # Initialize data using the unified data manager
        logger.info("Starting data initialization using the unified data manager")
        success = await data_manager.ensure_data_ready(
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
        
        if success:
            logger.info("Background data initialization completed successfully")
        else:
            logger.warning("Background data initialization completed with warnings")
        
        # Mark operation complete
        await data_manager.state_manager.mark_operation_complete("background_initialization", success)
    except Exception as e:
        logger.error(f"Error during background data initialization: {e}", exc_info=True)
        if data_manager and data_manager.state_manager:
            await data_manager.state_manager.mark_operation_complete("background_initialization", False, str(e))

# API routes
@app.get("/")
async def root():
    """Extremely lightweight root endpoint for deployment health checks.
    This MUST complete as quickly as possible with no dependencies on any other services."""
    # Return absolute minimal response for fastest possible health check
    # No logging, no timestamp calculation, nothing else that could potentially fail
    return {"status": "ok", "ready": True}

@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to docs."""
    return RedirectResponse(url="/docs")

@app.get("/healthz")
async def healthz():
    """Simple health check endpoint for Replit's health check system."""
    return {
        "status": "ok", 
        "ready": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/trigger-data-processing")
async def trigger_data_processing():
    """Manually trigger data processing."""
    logger.info("Manual data processing triggered via endpoint")
    
    # Create a background task
    task = asyncio.create_task(initialize_background_data())
    task.set_name("manual_data_init")
    logger.info(f"Created task from endpoint: {task.get_name()}")
    
    return {
        "status": "processing_started",
        "message": "Data processing has been triggered",
        "time": datetime.now().isoformat()
    }

@app.get("/api/v1/force-initialization")
async def force_initialization():
    """Force data initialization even if AUTO_CHECK_DATA is false."""
    logger.info("Manual data initialization triggered via endpoint")
    
    # Get current AUTO_CHECK_DATA setting to report it
    auto_check_data = os.environ.get('AUTO_CHECK_DATA', 'true').lower() in ('true', '1', 'yes')
    
    # Override environment variable temporarily
    original_value = os.environ.get('AUTO_CHECK_DATA')
    os.environ['AUTO_CHECK_DATA'] = 'true'
    
    # Log the values
    logger.info(f"Original AUTO_CHECK_DATA: {original_value}")
    logger.info(f"Original parsed value: {auto_check_data}")
    logger.info(f"Temporarily set AUTO_CHECK_DATA to: true")
    
    # Create a background task
    task = asyncio.create_task(initialize_background_data())
    task.set_name("manual_force_init")
    logger.info(f"Created initialization task: {task.get_name()}")
    
    # Restore original value if it was set
    if original_value is not None:
        os.environ['AUTO_CHECK_DATA'] = original_value
    else:
        del os.environ['AUTO_CHECK_DATA']
    
    return {
        "status": "initialization_started",
        "message": "Data initialization has been triggered",
        "original_auto_check_data": original_value,
        "original_parsed_value": auto_check_data,
        "time": datetime.now().isoformat()
    }

@app.get("/api/v1/initialization-status")
async def initialization_status():
    """Check initialization status."""
    # Get current environment variables
    auto_check_data = os.environ.get('AUTO_CHECK_DATA')
    force_refresh = os.environ.get('FORCE_DATA_REFRESH')
    skip_embeddings = os.environ.get('SKIP_EMBEDDINGS')
    
    # Check for data files
    data_config = chanscope_config.get_env_specific_attributes()
    
    # Get file paths
    complete_data_file = data_config.get('complete_data_file')
    stratified_file = data_config.get('stratified_file')
    embeddings_file = data_config.get('embeddings_file')
    thread_id_map_file = data_config.get('thread_id_map_file')
    
    # Check if files exist
    complete_data_exists = os.path.exists(complete_data_file) if complete_data_file else False
    stratified_exists = os.path.exists(stratified_file) if stratified_file else False
    embeddings_exist = os.path.exists(embeddings_file) if embeddings_file else False
    thread_map_exists = os.path.exists(thread_id_map_file) if thread_id_map_file else False
    
    return {
        "environment_variables": {
            "AUTO_CHECK_DATA": auto_check_data,
            "FORCE_DATA_REFRESH": force_refresh,
            "SKIP_EMBEDDINGS": skip_embeddings
        },
        "data_files": {
            "complete_data": {
                "path": str(complete_data_file),
                "exists": complete_data_exists
            },
            "stratified_sample": {
                "path": str(stratified_file),
                "exists": stratified_exists
            },
            "embeddings": {
                "path": str(embeddings_file),
                "exists": embeddings_exist
            },
            "thread_id_map": {
                "path": str(thread_id_map_file),
                "exists": thread_map_exists
            }
        },
        "ready": complete_data_exists and stratified_exists and (embeddings_exist or skip_embeddings == "true"),
        "time": datetime.now().isoformat()
    }

async def initialize_background_data():
    """Initialize data in background without blocking API startup."""
    try:
        logger.info("Background data initialization task started")
        logger.info(f"Environment variables at task start: AUTO_CHECK_DATA={os.environ.get('AUTO_CHECK_DATA')}, SKIP_EMBEDDINGS={os.environ.get('SKIP_EMBEDDINGS')}")
        
        # Add additional delay to ensure health checks have succeeded first
        # Increase the delay to give more time for health checks to complete
        logger.info("Waiting 20 seconds before starting data initialization...")
        await asyncio.sleep(20.0)  # Increased from 10.0 to 20.0 to ensure health checks pass
        
        # Get environment type from centralized function
        env_type = detect_environment()
        logger.info(f"Detected environment type: {env_type}")
        
        # Read environment variables again to ensure they haven't changed
        force_refresh = os.environ.get('FORCE_DATA_REFRESH', 'false').lower() in ('true', '1', 'yes')
        skip_embeddings = os.environ.get('SKIP_EMBEDDINGS', 'false').lower() in ('true', '1', 'yes')
        
        logger.info(f"Data initialization parameters: force_refresh={force_refresh}, skip_embeddings={skip_embeddings}")
        
        if env_type.lower() == 'replit':
            # For Replit, use database row count as a check
            logger.info("Running in Replit environment, checking database status")
            
            try:
                # Import and initialize PostgreSQL database connection
                from config.replit import PostgresDB
                db = PostgresDB()
                row_count = db.get_row_count()
                logger.info(f"Database row count: {row_count}")
                
                if row_count == 0:
                    logger.info("Database is empty, setting force_refresh=True")
                    force_refresh = True
            except Exception as e:
                logger.error(f"Error checking database status: {e}")
                # Continue with initialization
        else:
            # For Docker environment, check data files directly
            logger.info("Running in Docker environment, checking data files")
            
            # Check if the data directory has files
            # Use get_env_specific_attributes() method instead of direct attribute access
            env_specific = data_manager.config.get_env_specific_attributes()
            complete_data_path = env_specific.get('complete_data_file')
            stratified_path = data_manager.config.stratified_data_path
            
            if complete_data_path and os.path.exists(str(complete_data_path)):
                logger.info(f"Complete data file exists: {complete_data_path}")
            else:
                logger.info(f"Complete data file not found, will need to create it")
                force_refresh = True
                
            stratified_file = os.path.join(stratified_path, 'stratified_sample.csv')
            if os.path.exists(stratified_file):
                logger.info(f"Stratified file exists: {stratified_file}")
            else:
                logger.info(f"Stratified file not found, will need to create it")
                
            embeddings_file = os.path.join(stratified_path, 'embeddings.npz')
            if os.path.exists(embeddings_file):
                logger.info(f"Embeddings file exists: {embeddings_file}")
            else:
                logger.info(f"Embeddings file not found, will need to create it")
        
        # Use the unified data manager approach to ensure data is ready
        logger.info(f"Starting data preparation with force_refresh={force_refresh}, skip_embeddings={skip_embeddings}")
        
        # Mark initialization in progress
        await data_manager.state_manager.mark_operation_start("background_initialization")
        
        # Ensure data is ready
        success = await data_manager.ensure_data_ready(
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
        
        # Mark initialization complete
        if success:
            await data_manager.state_manager.mark_operation_complete(
                "background_initialization",
                {"status": "success", "timestamp": datetime.now().isoformat()}
            )
            logger.info("Data preparation completed successfully")
            
            # Create a task to periodically check data freshness if enabled
            refresh_interval = int(os.environ.get('DATA_REFRESH_INTERVAL', '86400'))  # Default to daily
            if os.environ.get('AUTO_REFRESH_DATA', 'false').lower() in ('true', '1', 'yes') and refresh_interval > 0:
                logger.info(f"Setting up periodic data refresh every {refresh_interval} seconds")
                asyncio.create_task(periodic_refresh(refresh_interval))
        else:
            await data_manager.state_manager.mark_operation_complete(
                "background_initialization",
                {"status": "error", "timestamp": datetime.now().isoformat()}
            )
            logger.error("Data preparation failed")
            
        return success
    except Exception as e:
        logger.error(f"Error during background data initialization: {e}")
        # Try to log the full stack trace for debugging
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to mark initialization as failed
        try:
            await data_manager.state_manager.mark_operation_complete(
                "background_initialization",
                {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
            )
        except Exception as mark_error:
            logger.error(f"Error marking operation as failed: {mark_error}")
            
        return False

# Run server directly if module is executed
if __name__ == "__main__":
    import uvicorn
    
    # Get port and host from Config instead of direct os.getenv
    api_settings = Config.get_api_settings()
    port = api_settings['port']
    host = api_settings['host']
    
    # Ensure we're listening on the correct port for Replit deployments
    if is_replit_environment():
        logger.info("Running in Replit environment, using port 80 and host 0.0.0.0")
        port = 80
        host = '0.0.0.0'
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Run server with optimized settings for Replit deployment
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        timeout_keep_alive=120,  # Longer keep-alive timeout for stable connections
        access_log=True
    )

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app.
    
    Returns:
        FastAPI: The configured FastAPI application instance
    """
    return app