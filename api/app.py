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
from config.env_loader import load_environment, detect_environment
from config.logging_config import setup_logging

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
        
        if auto_check_data:
            logger.info("AUTO_CHECK_DATA is enabled, initiating data preparation in background")
            # Create the task but don't await it - let it run fully in background
            asyncio.create_task(initialize_background_data())
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

async def initialize_background_data():
    """Initialize data in background without blocking API startup."""
    try:
        # Add additional delay to ensure health checks have succeeded first
        # Increase the delay to give more time for health checks to complete
        await asyncio.sleep(10.0)  # Ensure we've had enough time to pass initial healthchecks
        
        # Get environment type from centralized function
        env_type = detect_environment()
        force_refresh = os.environ.get('FORCE_DATA_REFRESH', 'false').lower() in ('true', '1', 'yes')
        skip_embeddings = os.environ.get('SKIP_EMBEDDINGS', 'false').lower() in ('true', '1', 'yes')
        
        # Log startup delay for Replit environment in particular
        if env_type == 'replit':
            logger.info("Running in Replit environment, using extended initialization sequence")
            
            # Add an additional delay for Replit deployments to ensure health checks succeed
            logger.info("Waiting additional time before starting heavy operations...")
            await asyncio.sleep(5.0)
            
            # Only do one small operation at a time with yields to allow health checks to complete
            
            # Initialize PostgreSQL schema if needed - lightweight operation
            from config.replit import PostgresDB
            try:
                logger.info("Step 1/4: Initializing PostgreSQL schema")
                db = PostgresDB()
                # Check if schema needs initialization
                db.initialize_schema()
                logger.info("PostgreSQL schema verified/initialized")
                
                # Yield control to allow other operations
                await asyncio.sleep(0.1)
                
                logger.info("Step 2/4: Checking data status")
                # Check if database is empty and we need to load data
                row_count = await data_manager.complete_data_storage.get_row_count()
                logger.info(f"PostgreSQL database has {row_count} rows")
                
                # Yield control to allow other operations
                await asyncio.sleep(0.1)
                
                if row_count == 0:
                    logger.info("PostgreSQL database is empty, force-refreshing data")
                    # Force refresh to ensure data is loaded
                    force_refresh = True
                else:
                    # Check if data is already fresh
                    logger.info("Step 3/4: Checking data freshness")
                    is_fresh = await data_manager.complete_data_storage.is_data_fresh()
                    if is_fresh:
                        logger.info(f"Database already contains {row_count} rows and data is fresh. Skipping data preparation.")
                        # Check if stratified sample exists
                        strat_exists = await data_manager.stratified_storage.sample_exists()
                        if strat_exists:
                            # Check if embeddings exist
                            embeddings_exist = await data_manager.embedding_storage.embeddings_exist()
                            if embeddings_exist or skip_embeddings:
                                logger.info("All required data components already exist. No data preparation needed.")
                                return
                            else:
                                logger.info("Embeddings missing, will only generate embeddings")
                                skip_embeddings = False
                                force_refresh = False
                        else:
                            logger.info("Stratified sample missing, will prepare stratified data")
                            # Don't need to refresh complete data
                            force_refresh = False
                
                logger.info("Step 4/4: Ensuring data is ready. This may take several minutes.")
            except Exception as e:
                logger.error(f"Error initializing PostgreSQL schema: {e}")
        
        # Use the unified data manager approach to ensure data is ready
        logger.info(f"Starting data preparation with force_refresh={force_refresh}, skip_embeddings={skip_embeddings}")
        success = await data_manager.ensure_data_ready(force_refresh=force_refresh, skip_embeddings=skip_embeddings)
        if success:
            logger.info("Data preparation completed successfully via the unified data manager")
        else:
            logger.warning("Data preparation completed with warnings")
    except Exception as e:
        logger.error(f"Error during background data initialization: {e}", exc_info=True)

# Run server directly if module is executed
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get('API_PORT', 80))
    host = os.environ.get('API_HOST', '0.0.0.0')
    
    # Ensure we're listening on the correct port for Replit deployments
    if os.environ.get('REPLIT_ENV'):
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