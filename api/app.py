"""FastAPI application configuration and middleware."""
import os
from typing import Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import time
import traceback
import json
from datetime import datetime
from .routes import router as api_router
from .exceptions import BaseAppException  # Updated import
from config.logging_config import get_logger
from . import get_environment  # Import from __init__.py
from config.env_loader import is_replit_environment, configure_replit_environment
from knowledge_agents.data_ops import DataOperations, DataConfig

logger = get_logger(__name__)

# Performance monitoring middleware
def add_middleware(app: FastAPI) -> None:
    """Add middleware to FastAPI app."""
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next: Any) -> Any:
        """Add processing time to response headers."""
        try:
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            logger.error(f"Error in process time middleware: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error in middleware"}
            )

    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next: Any) -> Any:
        """Global error handling middleware."""
        try:
            return await call_next(request)
        except BaseAppException as e:  # Updated to use BaseAppException
            logger.error(f"API error: {str(e)}")
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_response().dict()  # Use the standardized response format
            )
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}")
            logger.debug(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Clean up resources on shutdown."""
        try:
            logger.info("Shutting down Knowledge Agents API")
            # Add any cleanup code here
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            logger.debug(traceback.format_exc())

async def initialize_data_in_background(data_config: DataConfig):
    """Initialize data in background without blocking API startup.
    
    This implementation uses improved locking mechanisms to ensure
    only one worker performs initialization.
    """
    worker_id = os.getenv("WORKER_ID", str(os.getpid()))
    logger.info(f"Worker {worker_id} starting data initialization")
    
    try:
        # Create proper directory structure if needed
        data_dir = data_config.root_data_path
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use more robust initialization markers
        initialization_marker = data_dir / '.initialization_in_progress'
        completion_marker = data_dir / '.initialization_complete'
        state_file = data_dir / '.initialization_state'
        
        # Check if initialization is already complete with fresh data
        if completion_marker.exists() and state_file.exists():
            try:
                state_age = time.time() - state_file.stat().st_mtime
                if state_age < 3600:  # Less than 1 hour old
                    logger.info(f"Recent initialization found (age: {state_age:.1f}s), skipping")
                    return
            except Exception as e:
                logger.warning(f"Error checking initialization state age: {e}")
                
        # Create marker to prevent duplicate initialization
        with open(initialization_marker, "w") as f:
            json.dump({
                "start": datetime.now().isoformat(),
                "pid": os.getpid(),
                "worker_id": worker_id
            }, f)

        # Initialize data operations
        data_ops = DataOperations(data_config)

        # Run data initialization
        await data_ops.ensure_data_ready(force_refresh=True, skip_embeddings=False)

        # Create completion marker and state file when done
        with open(completion_marker, "w") as f:
            f.write(datetime.now().isoformat())
            
        # Write detailed state information
        with open(state_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "worker_id": worker_id,
                "pid": os.getpid(),
                "data_path": str(data_config.root_data_path)
            }, f)

        # Remove in-progress marker when done
        if initialization_marker.exists():
            initialization_marker.unlink()

        logger.info("Background data initialization completed successfully")
    except Exception as e:
        logger.error(f"Error in background data initialization: {e}", exc_info=True)
        # Make sure to clean up the initialization marker on error
        try:
            if initialization_marker.exists():
                initialization_marker.unlink()
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up initialization marker: {cleanup_error}")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Configure Replit environment if needed
    if is_replit_environment():
        configure_replit_environment()

    # Get port configuration
    port = int(os.getenv("PORT", "80"))
    host = os.getenv("HOST", "0.0.0.0")

    # Log startup configuration
    logger.info(f"Starting application on {host}:{port}")
    logger.info(f"Environment: {get_environment()}")
    logger.info(f"Replit mode: {is_replit_environment()}")

    # Create FastAPI app with enhanced metadata
    app = FastAPI(
        title="Knowledge Agent API",
        description="API for processing and analyzing text using AI models",
        version="1.0.0",
        docs_url="/docs",  # Always enable docs
        redoc_url="/redoc"  # Always enable redoc
    )

    # Add middleware
    add_middleware(app)

    # Include API routes with proper prefix
    app.include_router(api_router, prefix="/api/v1", tags=["knowledge_agents"])

    # Add root redirect to docs
    @app.get("/")
    async def root_redirect():
        """Redirect root to docs."""
        return RedirectResponse(url="/docs")

    # Add health check endpoint
    @app.get("/healthz")
    async def healthz():
        """Simple health check endpoint."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "port": port,
            "host": host
        }

    return app

# Create the FastAPI application instance
app = create_app()

# Function to get the app instance for module-level imports
def get_app() -> FastAPI:
    """Get the FastAPI application instance.
    
    Returns:
        The configured FastAPI application
    """
    return app

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn

    # Get port configuration
    port = int(os.getenv("PORT", "80"))
    host = os.getenv("HOST", "0.0.0.0")

    # Run the application
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=get_environment() == "development"
    )