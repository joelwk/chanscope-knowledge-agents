"""FastAPI application configuration and middleware."""
import os
from typing import Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import time
import traceback

from config.settings import Config
from .routes import router as api_router, APIError
from config.logging_config import get_logger
from . import get_environment  # Import from __init__.py

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
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": str(e)}
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

    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize necessary components on startup."""
        from . import initialize_data_processing  # Import here to avoid circular dependency
        
        logger.info("Starting Knowledge Agents API")
        
        # Log configuration details once
        config = Config.get_paths()
        api_settings = Config.get_api_settings()
        
        logger.info("API Configuration:")
        logger.info(f"- Environment: {get_environment()}")
        logger.info(f"- Docker: {api_settings.get('docker_env', False)}")
        logger.info(f"- Debug: {api_settings.get('debug', False)}")
        
        # Log paths
        for key, value in config.items():
            logger.info(f"- {key}: {value}")
        
        try:
            logger.info("Initializing data processing...")
            await initialize_data_processing()
            logger.info("Application startup complete")
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            raise

# Create application instance based on environment
def get_app() -> FastAPI:
    """Get the appropriate FastAPI application instance based on environment."""
    # Check if we're in Replit environment
    is_replit = os.getenv("REPLIT_ENV") in ["true", "replit", "production"]
    
    if is_replit:
        from . import create_replit_app
        logger.info("Creating Replit-specific FastAPI application")
        return create_replit_app()
    else:
        from . import create_app
        logger.info("Creating standard FastAPI application")
        return create_app()

# Default application instance
app = get_app()

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    
    # Get port configuration
    port = int(os.getenv("PORT", "80"))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Log startup information
    logger.info(f"Starting application on {host}:{port}")
    logger.info(f"Environment: {get_environment()}")
    logger.info(f"Replit mode: {os.getenv('REPLIT_ENV', 'false')}")
    
    # Run the application
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        log_level="info",
        reload=get_environment() == "development"
    )