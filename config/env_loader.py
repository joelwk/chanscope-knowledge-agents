"""Environment loader module."""
from dotenv import load_dotenv
import os
import logging
from config.base_settings import get_env_path
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from root .env file.
    
    This enhanced loader supports section headers in the .env file:
    [docker], [replit], and [local] sections will be loaded based on
    the detected environment.
    
    Raises:
        FileNotFoundError: If .env file doesn't exist at the expected location.
    """
    env_path = get_env_path()
    logger.info(f"Attempting to load environment from: {env_path}")
    
    if not env_path.exists():
        logger.error(f"Could not find .env file at {env_path}")
        raise FileNotFoundError(f"Could not find .env file at {env_path}")
    
    # First load the base environment variables (before any section)
    load_dotenv(dotenv_path=env_path)
    
    # Then load the appropriate section based on the environment
    load_environment_section(env_path)
    
    # Validate critical environment variables
    _validate_critical_vars()

def load_environment_section(env_path):
    """Load environment variables from the appropriate section in the .env file."""
    # Determine which section to load
    section = None
    if is_docker_environment():
        section = "docker"
        logger.info("Detected Docker environment, loading [docker] section")
    elif is_replit_environment():
        section = "replit"
        logger.info("Detected Replit environment, loading [replit] section")
    else:
        section = "local"
        logger.info("Detected local environment, loading [local] section")
    
    # Read the .env file and extract the appropriate section
    try:
        with open(env_path, 'r') as f:
            content = f.read()
        
        # Find the section
        section_pattern = rf"\[{section}\](.*?)(\[|$)"
        match = re.search(section_pattern, content, re.DOTALL)
        
        if match:
            section_content = match.group(1).strip()
            logger.info(f"Found [{section}] section with {len(section_content.splitlines())} lines")
            
            # Parse the section content and set environment variables
            for line in section_content.splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Only set if not already set or override is needed
                    if key not in os.environ or section != "local":  # Always override with non-local sections
                        os.environ[key] = value
                        logger.debug(f"Set environment variable from [{section}] section: {key}")
        else:
            logger.warning(f"Could not find [{section}] section in .env file")
    
    except Exception as e:
        logger.error(f"Error loading environment section: {str(e)}")

def is_docker_environment() -> bool:
    """Detect if running in Docker environment."""
    return os.environ.get("DOCKER_ENV", "").lower() == "true" or \
           os.path.exists("/.dockerenv") or \
           os.path.exists("/proc/1/cgroup") and "docker" in open("/proc/1/cgroup").read()

def _validate_critical_vars():
    """Validate that critical environment variables are set."""
    critical_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for model access',
        'AWS_ACCESS_KEY_ID': 'AWS access key for S3 storage',
        'AWS_SECRET_ACCESS_KEY': 'AWS secret key for S3 storage',
        'ROOT_DATA_PATH': 'Root data path',
        'STRATIFIED_PATH': 'Stratified data path',
        'PATH_TEMP': 'Temporary files path'
    }
    
    for var, description in critical_vars.items():
        value = os.getenv(var)
        if value:
            # Log securely - only show first/last few chars for sensitive data
            if 'KEY' in var or 'SECRET' in var:
                logger.info(f"{var} loaded: {value[:5]}...{value[-5:]}")
            else:
                logger.info(f"{var} loaded: {value}")
        else:
            logger.warning(f"Missing {description} ({var})")

def is_replit_environment() -> bool:
    """Detect if running in Replit environment."""
    return os.environ.get("REPLIT_ENV") == "replit" or \
           os.environ.get("REPLIT_DEPLOYMENT") == "1" or \
           "REPLIT_ID" in os.environ

def get_replit_paths() -> dict:
    """Get Replit-specific paths."""
    repl_home = os.environ.get("REPL_HOME", "")
    return {
        "root_data_path": os.path.join(repl_home, "data"),
        "stratified_path": os.path.join(repl_home, "data/stratified"),
        "temp_path": os.path.join(repl_home, "temp_files"),
        "logs_path": os.path.join(repl_home, "logs")
    }

def configure_replit_environment():
    """Configure environment based on detection."""
    if is_replit_environment():
        paths = get_replit_paths()
        # Create directories
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ.update({
            "ROOT_DATA_PATH": paths["root_data_path"],
            "STRATIFIED_PATH": paths["stratified_path"],
            "PATH_TEMP": paths["temp_path"],
            "EMBEDDING_BATCH_SIZE": os.environ.get("EMBEDDING_BATCH_SIZE", "10"),
            "CHUNK_BATCH_SIZE": os.environ.get("CHUNK_BATCH_SIZE", "10"),
            "PROCESSING_CHUNK_SIZE": os.environ.get("PROCESSING_CHUNK_SIZE", "5000"),
            "MAX_WORKERS": os.environ.get("MAX_WORKERS", "2")
        })

def configure_docker_environment():
    """Configure environment variables for Docker deployment."""
    if is_docker_environment():
        logger.info("Configuring Docker-specific environment variables")
        
        # Set Docker-specific paths if not already set
        docker_data_path = os.environ.get("ROOT_DATA_PATH", "/app/data")
        docker_stratified_path = os.environ.get("STRATIFIED_PATH", "/app/data/stratified")
        docker_temp_path = os.environ.get("PATH_TEMP", "/app/temp_files")
        
        # Create directories
        for path in [docker_data_path, docker_stratified_path, docker_temp_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Set environment variables with Docker-specific defaults
        os.environ.update({
            "ROOT_DATA_PATH": docker_data_path,
            "STRATIFIED_PATH": docker_stratified_path,
            "PATH_TEMP": docker_temp_path,
            "DOCKER_ENV": "true",
            "EMBEDDING_BATCH_SIZE": os.environ.get("EMBEDDING_BATCH_SIZE", "20"),
            "CHUNK_BATCH_SIZE": os.environ.get("CHUNK_BATCH_SIZE", "20"),
            "PROCESSING_CHUNK_SIZE": os.environ.get("PROCESSING_CHUNK_SIZE", "10000"),
            "MAX_WORKERS": os.environ.get("MAX_WORKERS", "4"),
            "DATA_UPDATE_INTERVAL": os.environ.get("DATA_UPDATE_INTERVAL", "3600")
        })
        
        logger.info(f"Docker environment configured with ROOT_DATA_PATH={docker_data_path}")

# Load environment variables when module is imported
load_environment() 