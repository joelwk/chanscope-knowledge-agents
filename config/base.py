"""Base configuration module without dependencies."""
import os
from pathlib import Path
from typing import Dict, Any
from config.base_settings import get_base_settings, get_base_paths, ensure_base_paths as _ensure_base_paths

def get_base_paths() -> Dict[str, Path]:
    """Get base paths without any dependencies."""
    return get_base_paths()

def ensure_base_paths() -> None:
    """Ensure base paths exist."""
    _ensure_base_paths()

class BaseConfig:
    """Base configuration without dependencies."""
    
    @staticmethod
    def get_env_path() -> Path:
        """Get the path to the .env file."""
        return get_base_paths()['root'] / '.env'
    
    @staticmethod
    def get_base_settings() -> Dict[str, Any]:
        """Get base settings from environment variables."""
        return get_base_settings()

# Initialize paths on module import
ensure_base_paths() 