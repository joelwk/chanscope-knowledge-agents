"""Base configuration module for path and environment setup."""
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Environment file path
ENV_PATH = ROOT_DIR / '.env'

# Basic path configuration
def get_base_paths():
    """Get base paths configuration."""
    return {
        "root": ROOT_DIR,
        "env": ENV_PATH,
        "data": ROOT_DIR / 'data',
        "logs": ROOT_DIR / 'logs',
        "temp": ROOT_DIR / 'temp_files'
    }

def get_directory_paths():
    """Get only the directory paths that should be created."""
    paths = get_base_paths()
    # Filter out file paths (env) and special paths (root)
    return {k: v for k, v in paths.items() 
            if k not in ('env', 'root')}

# Ensure critical directories exist
def ensure_base_paths():
    """Create base directories if they don't exist."""
    directory_paths = get_directory_paths()
    
    for path_name, path in directory_paths.items():
        try:
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path}")
        except Exception as e:
            logger.error(f"Error creating directory {path_name} at {path}: {str(e)}")
            raise 