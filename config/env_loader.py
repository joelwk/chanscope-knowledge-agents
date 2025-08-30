"""Environment loader module."""
from dotenv import load_dotenv
import os
import logging
from config.base_settings import get_env_path
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def detect_environment() -> str:
    """
    Detect the current execution environment.
    
    This is the centralized function for detecting whether we're running in
    Replit, Docker, or local environment. All code should use this function
    instead of implementing their own detection logic.
    
    Returns:
        str: 'replit' if running in Replit, 'docker' if running in Docker, 
             or 'docker' as default/fallback (considering local as equivalent to Docker)
    """
    # Check Replit environment first - must have REPL_ID or explicit REPLIT_ENV=replit
    replit_env = os.environ.get('REPLIT_ENV', '').lower()
    has_repl_id = os.environ.get('REPL_ID') is not None
    
    if (replit_env in ('replit', 'true')) or has_repl_id:
        return 'replit'
    
    # Check Docker environment - prioritize file system detection over environment vars
    # This ensures Docker containers are always detected correctly regardless of .env file content
    if os.path.exists('/.dockerenv'):
        return 'docker'
    elif os.environ.get('DOCKER_ENV', '').lower() == 'true':
        return 'docker'
    elif os.environ.get('ENVIRONMENT', '').lower() == 'docker':
        return 'docker'
    
    # Default to Docker (treating local as Docker-equivalent)
    else:
        return 'docker'

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
    
    # Base environment variables are already loaded by base_settings.py
    # Don't load_dotenv here as it would load ALL sections including conflicting variables
    
    # Detect environment before loading section
    is_replit = is_replit_environment()
    is_docker = is_docker_environment() if not is_replit else False
    
    # Log the detected environment
    if is_replit:
        logger.info("Detected Replit environment")
    elif is_docker:
        logger.info("Detected Docker environment")
    else:
        logger.info("Detected local environment")
    
    # Then load the appropriate section based on the environment
    load_environment_section(env_path)
    
    # Configure environment-specific settings
    if is_replit:
        logger.info("Configuring Replit-specific settings")
        try:
            configure_replit_environment()
        except Exception as e:
            logger.warning(f"Replit environment configuration skipped due to error: {e}")
    elif is_docker:
        logger.info("Configuring Docker-specific settings")
        try:
            configure_docker_environment()
        except Exception as e:
            logger.warning(f"Docker environment configuration skipped due to error: {e}")
    else:
        logger.info("Using local environment settings")
    
    # Validate critical environment variables
    _validate_critical_vars()

def load_environment_section(env_path):
    """Load environment variables from the appropriate section in the .env file."""
    # Determine which section to load
    section = None
    if is_replit_environment():
        section = "replit"
        logger.info("Detected Replit environment, loading [replit] section")
    elif is_docker_environment():
        section = "docker"
        logger.info("Detected Docker environment, loading [docker] section")
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
            
            # If Replit section is not found but we're in Replit, set default values
            if section == "replit":
                logger.info("Setting default Replit environment variables")
                os.environ.update({
                    "REPLIT_ENV": "replit",
                    "DOCKER_ENV": "false",
                    "USE_MOCK_DATA": os.environ.get("USE_MOCK_DATA", "false"),  # Default to false
                    "USE_MOCK_EMBEDDINGS": os.environ.get("USE_MOCK_EMBEDDINGS", "false"),  # Default to false
                    "DATA_RETENTION_DAYS": os.environ.get("DATA_RETENTION_DAYS", "14"),
                    "FILTER_DATE": os.environ.get("FILTER_DATE", "")
                })
                logger.info(f"Default Replit environment variables set with DATA_RETENTION_DAYS={os.environ.get('DATA_RETENTION_DAYS')} and FILTER_DATE={os.environ.get('FILTER_DATE')}")
                logger.info(f"Mock data is {'enabled' if os.environ.get('USE_MOCK_DATA', '').lower() == 'true' else 'disabled'} in Replit environment")
    
    except Exception as e:
        logger.error(f"Error loading environment section: {str(e)}")

def is_docker_environment() -> bool:
    """
    Check if running in Docker environment.
    
    Returns:
        bool: True if running in Docker (or local), False otherwise
    """
    return detect_environment() == 'docker'

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
    """
    Check if running in Replit environment.
    
    Returns:
        bool: True if running in Replit, False otherwise
    """
    return detect_environment() == 'replit'

def get_replit_paths() -> dict:
    """Get Replit-specific paths."""
    repl_home = os.environ.get("REPL_HOME", "/home/runner/workspace")
    return {
        "root_data_path": os.path.join(repl_home, "data"),
        "stratified_path": os.path.join(repl_home, "data/stratified"),
        "generated_data_path": os.path.join(repl_home, "data/generated_data"),
        "temp_path": os.path.join(repl_home, "temp_files"),
        "logs_path": os.path.join(repl_home, "logs"),
        "mock_data_path": os.path.join(repl_home, "data/mock")
    }

def configure_replit_environment():
    """Configure environment based on detection."""
    if is_replit_environment():
        logger.info("Configuring Replit-specific environment variables")
        
        # Get Replit home directory
        repl_home = os.environ.get("REPL_HOME", "/home/runner/workspace")
        
        # Set up paths
        paths = get_replit_paths()
        
        # Create directories
        for path in paths.values():
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create path {path}: {e}")
        
        # Check if we're using mock data
        use_mock_data = os.environ.get("USE_MOCK_DATA", "").lower() == "true"
        
        # Set environment variables with Replit-specific defaults
        env_vars = {
            "ROOT_DATA_PATH": paths["root_data_path"],
            "STRATIFIED_PATH": paths["stratified_path"],
            "GENERATED_DATA_PATH": paths["generated_data_path"],
            "PATH_TEMP": paths["temp_path"],
            "REPLIT_ENV": "replit",
            "DOCKER_ENV": "false",
            "USE_MOCK_DATA": os.environ.get("USE_MOCK_DATA", "false"),  # Default to false
            "USE_MOCK_EMBEDDINGS": os.environ.get("USE_MOCK_EMBEDDINGS", "false"),  # Default to false
            "DATA_RETENTION_DAYS": os.environ.get("DATA_RETENTION_DAYS", "14"),
            "FILTER_DATE": os.environ.get("FILTER_DATE", ""),
            "EMBEDDING_BATCH_SIZE": os.environ.get("EMBEDDING_BATCH_SIZE", "10"),
            "CHUNK_BATCH_SIZE": os.environ.get("CHUNK_BATCH_SIZE", "10"),
            "PROCESSING_CHUNK_SIZE": os.environ.get("PROCESSING_CHUNK_SIZE", "5000"),
            "MAX_WORKERS": os.environ.get("MAX_WORKERS", "2"),
            "API_WORKERS": os.environ.get("API_WORKERS", "3")
        }
        
        # Update environment variables
        os.environ.update(env_vars)
        
        logger.info(f"Replit environment configured with ROOT_DATA_PATH={paths['root_data_path']}")
        logger.info(f"Data retention settings: DATA_RETENTION_DAYS={env_vars['DATA_RETENTION_DAYS']}, FILTER_DATE={env_vars['FILTER_DATE']}")
        if use_mock_data:
            logger.info("Mock data is enabled in Replit environment")
        else:
            logger.info("Mock data is disabled in Replit environment")

def configure_docker_environment():
    """Configure environment variables for Docker deployment."""
    if is_docker_environment():
        logger.info("Configuring Docker-specific environment variables")
        
        # Set Docker-specific paths if not already set
        docker_data_path = os.environ.get("ROOT_DATA_PATH", "/app/data")
        docker_stratified_path = os.environ.get("STRATIFIED_PATH", "/app/data/stratified")
        docker_generated_data_path = os.environ.get("GENERATED_DATA_PATH", "/app/data/generated_data")
        docker_temp_path = os.environ.get("PATH_TEMP", "/app/temp_files")
        
        # Create directories
        for path in [docker_data_path, docker_stratified_path, docker_generated_data_path, docker_temp_path]:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create path {path}: {e}")
        
        # Set environment variables with Docker-specific defaults
        os.environ.update({
            "ROOT_DATA_PATH": docker_data_path,
            "STRATIFIED_PATH": docker_stratified_path,
            "GENERATED_DATA_PATH": docker_generated_data_path,
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
