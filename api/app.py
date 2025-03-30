"""
Main API module for the Chanscope application.

This module provides the FastAPI application for the Chanscope approach,
supporting both file-based storage (Docker) and database storage (Replit).
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import time
from datetime import datetime

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

# Create FastAPI application
app = FastAPI(
    title="Chanscope API",
    description="API for the Chanscope data processing and query system.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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
    """Root endpoint."""
    return RedirectResponse(url="/docs")

@app.get("/healthz")
async def healthz():
    """Simple health check endpoint for Replit's health check system."""
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("REPLIT_ENV", "development")
    }

# Run on startup
@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("Starting Chanscope API...")
    
    # Initialize data in the background
    try:
        # Only initialize if specified in environment
        auto_check_data = os.environ.get('AUTO_CHECK_DATA', 'true').lower() in ('true', '1', 'yes')
        
        if auto_check_data:
            logger.info("AUTO_CHECK_DATA is enabled, initiating data preparation")
            # Get environment type from centralized function
            env_type = detect_environment()
            force_refresh = os.environ.get('FORCE_DATA_REFRESH', 'false').lower() in ('true', '1', 'yes')
            skip_embeddings = os.environ.get('SKIP_EMBEDDINGS', 'false').lower() in ('true', '1', 'yes')
            
            if env_type == 'replit':
                logger.info("Running in Replit environment, ensuring PostgreSQL schema is prepared")
                # Initialize PostgreSQL schema if needed
                from config.replit import PostgresDB
                try:
                    db = PostgresDB()
                    # Check if schema needs initialization
                    db.initialize_schema()
                    logger.info("PostgreSQL schema verified/initialized")
                    
                    # Check if database is empty and we need to load data
                    row_count = await data_manager.complete_data_storage.get_row_count()
                    if row_count == 0:
                        logger.info("PostgreSQL database is empty, force-refreshing data")
                        # Force refresh to ensure data is loaded
                        force_refresh = True
                except Exception as e:
                    logger.error(f"Error initializing PostgreSQL schema: {e}")
            
            # Use the unified data manager approach to ensure data is ready
            success = await data_manager.ensure_data_ready(force_refresh=force_refresh, skip_embeddings=skip_embeddings)
            if success:
                logger.info("Data preparation completed successfully via the unified data manager")
            else:
                logger.warning("Data preparation completed with warnings")
        else:
            logger.info("AUTO_CHECK_DATA is disabled, skipping initial data preparation")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

# Run server directly if module is executed
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get('API_PORT', 80))
    host = os.environ.get('API_HOST', '0.0.0.0')
    
    # Run server
    uvicorn.run(app, host=host, port=port)

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app.
    
    Returns:
        FastAPI: The configured FastAPI application instance
    """
    return app