"""Centralized logging configuration for the application."""
import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Any
import threading
import multiprocessing
from config.base_settings import get_base_settings, get_base_paths

# Thread-safe singleton lock
_logging_lock = threading.Lock()
_is_logging_configured = False

def setup_logging() -> None:
    """Configure logging with proper handlers and formatters in a thread-safe manner."""
    global _is_logging_configured
    
    with _logging_lock:
        if _is_logging_configured:
            return
        
        # Get settings from base_settings
        settings = get_base_settings()
        api_settings = settings.get('api', {})
        
        # Clean and normalize the log level to uppercase
        log_level = api_settings.get('log_level', 'INFO').strip().upper()
        
        # Get worker ID for log file naming
        worker_id = multiprocessing.current_process().name.replace('Process-', '')
        
        # Ensure log directory exists
        paths = get_base_paths()
        log_dir = paths.get('logs', Path('logs')).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_config: Dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": api_settings.get(
                        'log_format', 
                        f'%(asctime)s - [Worker-{worker_id}] - %(name)s - %(levelname)s - %(message)s'
                    ).strip(),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "filename": str(log_dir / f"app_worker_{worker_id}.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "filename": str(log_dir / f"error_worker_{worker_id}.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "level": "ERROR",
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "api": {
                    "handlers": ["console", "file", "error_file"],
                    "level": log_level,
                    "propagate": False,
                },
                "knowledge_agents": {
                    "handlers": ["console", "file", "error_file"],
                    "level": log_level,
                    "propagate": False,
                },
                "tests": {
                    "handlers": ["console", "file", "error_file"],
                    "level": log_level,
                    "propagate": False,
                }
            },
        }
        
        try:
            # Remove any existing handlers from root logger
            root_logger = logging.getLogger()
            if root_logger.handlers:
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)
            
            # Apply logging configuration
            logging.config.dictConfig(log_config)
            _is_logging_configured = True
            logging.info(f"Logging configuration applied successfully for worker {worker_id}")
            
            # Set up sensitive data filtering
            from config.logging_utils import setup_sensitive_data_filter
            setup_sensitive_data_filter()
            logging.info("Sensitive data filtering enabled for all loggers")
            
        except ValueError as e:
            # If there's an error with the logging configuration, set up a basic config
            logging.basicConfig(
                level=logging.INFO,
                format=f'%(asctime)s - [Worker-{worker_id}] - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(str(log_dir / f"fallback_worker_{worker_id}.log"))
                ]
            )
            logging.error(f"Error configuring logging: {str(e)}. Using basic configuration.")
            
            # Even with basic config, try to set up sensitive data filtering
            try:
                from config.logging_utils import setup_sensitive_data_filter
                setup_sensitive_data_filter()
                logging.info("Sensitive data filtering enabled with basic logging configuration")
            except Exception as filter_error:
                logging.error(f"Could not set up sensitive data filtering: {filter_error}")

def get_logger(name: str, level: str = None) -> logging.Logger:
    """Get a logger instance with the specified name and optional level.
    
    Args:
        name: The name of the logger
        level: Optional logging level to override the default
        
    Returns:
        A configured logger instance
    """
    if not _is_logging_configured:
        setup_logging()
    
    logger = logging.getLogger(name)
    
    # Set custom level if provided
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
        
    return logger 