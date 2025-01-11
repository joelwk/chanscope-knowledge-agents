"""Environment loader module."""
from dotenv import load_dotenv
import os
import logging
from config.base import ENV_PATH

logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from root .env file.
    
    Raises:
        FileNotFoundError: If .env file doesn't exist at the expected location.
    """
    logger.info(f"Attempting to load environment from: {ENV_PATH}")
    
    if not ENV_PATH.exists():
        logger.error(f"Could not find .env file at {ENV_PATH}")
        raise FileNotFoundError(f"Could not find .env file at {ENV_PATH}")
        
    load_dotenv(dotenv_path=ENV_PATH)
    
    # Validate critical environment variables
    _validate_critical_vars()

def _validate_critical_vars():
    """Validate that critical environment variables are set."""
    critical_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for model access',
        'FLASK_ENV': 'Flask environment (development/production)',
        'AWS_ACCESS_KEY_ID': 'AWS access key for S3 storage',
        'AWS_SECRET_ACCESS_KEY': 'AWS secret key for S3 storage'
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

# Load environment variables when module is imported
load_environment() 