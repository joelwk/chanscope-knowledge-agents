"""Configuration settings module."""
import os
import logging
from pathlib import Path
from config.base import ROOT_DIR, get_base_paths, ensure_base_paths

logger = logging.getLogger(__name__)

class Config:
    """Base configuration."""

    # Load base paths
    base_paths = get_base_paths()
    PROJECT_ROOT = base_paths['root']

    @staticmethod
    def _clean_api_key(key: str) -> str:
        """Clean API key by removing angle brackets and quotes."""
        if not key:
            logger.warning("API key is empty or None")
            return key
            
        key = key.strip()
        original_key = key  # Store original for comparison
        original_len = len(key)
        
        # Remove any hidden characters or whitespace
        key = ''.join(c for c in key if c.isprintable()).strip()
        
        if key.startswith("<"):
            key = key[1:]
        if key.endswith(">"):
            key = key[:-1]
        key = key.strip("'\"").strip()
        
        if len(key) != original_len:
            logger.info(f"API key was cleaned (length changed from {original_len} to {len(key)})")
            if original_key.startswith("AKIA"):  # AWS Key specific logging
                logger.info(f"AWS Key cleaning: Original prefix: {original_key[:7]}, New prefix: {key[:7]}")
                logger.info(f"AWS Key contains hidden characters: {'Yes' if len(original_key.encode()) != len(original_key) else 'No'}")
            
        if key.startswith("AKIA"):  # AWS specific validation
            if len(key) != 20:  # AWS Access Keys are 20 characters
                logger.warning(f"AWS Access Key has incorrect length: {len(key)}, expected 20")
            # Ensure only allowed characters
            if not all(c.isalnum() for c in key):
                logger.warning("AWS Access Key contains non-alphanumeric characters")
            
        return key

    @staticmethod
    def _parse_none_value(value: str) -> str | None:
        """Parse a string value that might represent None."""
        if not value or value.lower() == 'none':
            return None
        return value.strip()

    @staticmethod
    def _validate_model_name(model_name: str, provider: str,
                             model_type: str) -> str:
        """Validate and clean model name based on provider and type.
        
        Args:
            model_name: The model name to validate
            provider: The provider (openai, grok, venice)
            model_type: The type of model (completion, embedding, chunk)
        """
        if not model_name:
            logger.warning(
                f"Model name for {provider} {model_type} is empty or None")
            return model_name

        model_name = model_name.strip().strip("'\"").strip()

        # Provider-specific validation
        if provider == 'openai':
            valid_prefixes = {
                'completion': ['gpt-'],
                'embedding': ['text-embedding-'],
                'chunk': ['gpt-']
            }
            if model_type in valid_prefixes:
                if not any(
                        model_name.startswith(prefix)
                        for prefix in valid_prefixes[model_type]):
                    logger.warning(
                        f"OpenAI {model_type} model '{model_name}' may be invalid - doesn't start with expected prefix"
                    )

        elif provider == 'grok':
            valid_prefixes = {
                'completion': ['grok-'],
                'embedding': ['grok-'],
                'chunk': ['grok-']
            }
            if model_type in valid_prefixes:
                if not any(
                        model_name.startswith(prefix)
                        for prefix in valid_prefixes[model_type]):
                    logger.warning(
                        f"Grok {model_type} model '{model_name}' may be invalid - doesn't start with expected prefix"
                    )

        elif provider == 'venice':
            # Venice models have more varied names, just log the model being used
            logger.info(f"Using Venice {model_type} model: {model_name}")

        else:
            logger.warning(
                f"Unknown provider '{provider}' for model validation")

        return model_name

    # Data paths - loaded from environment with defaults
    ROOT_PATH = os.getenv('ROOT_PATH', 'data')
    DATA_PATH = os.getenv('DATA_PATH', 'data')
    ALL_DATA = os.getenv('ALL_DATA', 'data/all_data.csv')
    ALL_DATA_STRATIFIED_PATH = os.getenv('ALL_DATA_STRATIFIED_PATH', 'data/stratified')
    KNOWLEDGE_BASE = os.getenv('KNOWLEDGE_BASE', 'data/knowledge_base.csv')
    PATH_TEMP = os.getenv('PATH_TEMP', 'temp_files')

    DOCKER_ENV = os.getenv('DOCKER_ENV')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if DOCKER_ENV == 'true':
        OPENAI_API_KEY = _clean_api_key(OPENAI_API_KEY) 
        logger.info(
            f"Cleaned OPENAI_API_KEY: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}"
            )
    else:
        logger.info("Not running in Docker environment, using API key as-is")
        
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set or was cleaned to empty")

    # QUART settings
    QUART_APP = 'api.app'
    QUART_ENV = os.getenv('QUART_ENV')

    # API settings
    SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE'))
    MAX_WORKERS = int(os.getenv('MAX_WORKERS'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'

    # Model settings
    EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER','openai')
    CHUNK_PROVIDER = os.getenv('CHUNK_PROVIDER', 'openai')
    SUMMARY_PROVIDER = os.getenv('SUMMARY_PROVIDER', 'openai')
    
    # Default providers (used as fallbacks)
    DEFAULT_EMBEDDING_PROVIDER = 'openai'
    DEFAULT_CHUNK_PROVIDER = 'openai'  
    DEFAULT_SUMMARY_PROVIDER = 'openai'

    # OpenAI settings with model validation
    OPENAI_MODEL = _validate_model_name(os.getenv('OPENAI_MODEL', 'gpt-4o'),'openai', 'completion')
    OPENAI_EMBEDDING_MODEL = _validate_model_name(os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),'openai', 'embedding')

    # Grok settings with model validation
    GROK_API_KEY = _clean_api_key(os.getenv('GROK_API_KEY'))
    GROK_MODEL = _validate_model_name(os.getenv('GROK_MODEL', 'grok-2-1212'), 'grok', 'completion')

    # Venice settings with model validation
    VENICE_API_KEY = _clean_api_key(os.getenv('VENICE_API_KEY'))
    VENICE_MODEL = _validate_model_name(os.getenv('VENICE_MODEL', 'llama-3.1-405b'), 'venice', 'completion')
    VENICE_CHUNK_MODEL = _validate_model_name(os.getenv('VENICE_CHUNK_MODEL', 'dolphin-2.9.2-qwen2-72b'), 'venice','chunk')

    # AWS settings
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET', 'rolling-data')
    S3_BUCKET_PREFIX = os.getenv('S3_BUCKET_PREFIX', 'data')

    # Data processing settings
    TIME_COLUMN = os.getenv('TIME_COLUMN', 'posted_date_time')
    STRATA_COLUMN = os.getenv('STRATA_COLUMN')
    FREQ = os.getenv('FREQ', 'H')
    FILTER_DATE = os.getenv('FILTER_DATE')
    SELECT_BOARD = _parse_none_value(os.getenv('SELECT_BOARD'))
    PADDING_ENABLED = os.getenv('PADDING_ENABLED', 'false')
    CONTRACTION_MAPPING_ENABLED = os.getenv('CONTRACTION_MAPPING_ENABLED','false')
    NON_ALPHA_NUMERIC_ENABLED = os.getenv('NON_ALPHA_NUMERIC_ENABLED', 'false')

    # API Configuration
    API_HOST = os.getenv('API_HOST') 
    # Use Replit's PORT env var if available, otherwise use API_PORT from env or default to 5000
    API_PORT = os.getenv('API_PORT')
    API_TIMEOUT = 1500

    # Batch size settings with appropriate defaults
    EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE'))
    CHUNK_BATCH_SIZE = int(os.getenv('CHUNK_BATCH_SIZE'))
    SUMMARY_BATCH_SIZE = int(os.getenv('SUMMARY_BATCH_SIZE'))

    @classmethod
    def _get_base_urls(cls):
        """Get the base URLs with proper fallback for different environments."""
        urls = []
        
        # Check environment
        is_docker = os.getenv('DOCKER_ENV') == 'true'
        is_replit = bool(os.getenv('REPL_SLUG')) and bool(os.getenv('REPL_OWNER'))
        
        if is_docker:
            # In Docker, use the service name as host
            docker_host = os.getenv('API_HOST', 'api')
            docker_port = int(os.getenv('API_PORT', '5000'))
            urls.append(f"http://{docker_host}:{docker_port}")
            logger.info(f"Running in Docker environment, using URL: {urls[0]}")
        elif is_replit:
            # Add Replit URL
            repl_slug = os.getenv('REPL_SLUG')
            repl_owner = os.getenv('REPL_OWNER')
            urls.append(f"https://{repl_slug}.{repl_owner}.repl.co")
            logger.info(f"Running in Replit environment, using URL: {urls[0]}")
        
        # Always add local URL as fallback
        local_url = f"http://{cls.API_HOST}:{cls.API_PORT}"
        if local_url not in urls:
            urls.append(local_url)
        
        logger.info(f"Generated API URLs: {urls}")
        return urls

    @classmethod
    def get_api_settings(cls):
        """Get API-related settings."""
        urls = cls._get_base_urls()
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "base_urls": urls,
            "base_url": urls[0],  # Primary URL
            "timeout": cls.API_TIMEOUT,
        }
        
    @classmethod
    def get_provider_settings(cls):
        """Get provider settings with defaults."""
        return {
            "embedding_provider": cls.EMBEDDING_PROVIDER,
            "chunk_provider": cls.CHUNK_PROVIDER,
            "summary_provider": cls.SUMMARY_PROVIDER,
        }
    @classmethod
    def get_data_paths(cls):
        """Get all data-related paths."""
        return {
            "root": cls.ROOT_PATH,
            "data": cls.DATA_PATH,
            "all_data": cls.ALL_DATA,
            "stratified": cls.ALL_DATA_STRATIFIED_PATH,
            "knowledge_base": cls.KNOWLEDGE_BASE,
            "temp": cls.PATH_TEMP,
        }

    @classmethod
    def get_processing_settings(cls):
        """Get data processing settings."""
        return {
            "sample_size": cls.SAMPLE_SIZE,
            "max_workers": cls.MAX_WORKERS,
            "max_tokens": cls.MAX_TOKENS,
            "chunk_size": cls.CHUNK_SIZE,
            "cache_enabled": cls.CACHE_ENABLED,
            "time_column": cls.TIME_COLUMN,
            "strata_column": cls.STRATA_COLUMN,
            "freq": cls.FREQ,
            "filter_date": cls.FILTER_DATE,
            "padding_enabled": cls.PADDING_ENABLED,
            "contraction_mapping_enabled": cls.CONTRACTION_MAPPING_ENABLED,
            "non_alpha_numeric_enabled": cls.NON_ALPHA_NUMERIC_ENABLED,
        }

    @classmethod
    def validate_paths(cls):
        """Ensure all required paths exist"""
        # First ensure base paths exist
        ensure_base_paths()

        # Then ensure application-specific paths exist
        paths = [
            Path(cls.PROJECT_ROOT) / cls.ROOT_PATH,
            Path(cls.PROJECT_ROOT) / cls.DATA_PATH,
            Path(cls.PROJECT_ROOT) / cls.ALL_DATA_STRATIFIED_PATH,
            Path(cls.PROJECT_ROOT) / cls.PATH_TEMP
        ]

        for path in paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"Could not create directory {path}: {e}")
                # Continue even if directory creation fails
                pass

    @classmethod
    def initialize_paths(cls):
        """Initialize paths from environment variables if in Docker."""
        if os.getenv('DOCKER_ENV') == 'true':
            cls.ROOT_PATH = '/app/data'
            cls.DATA_PATH = '/app/data'
            cls.ALL_DATA = '/app/data/all_data.csv'
            cls.ALL_DATA_STRATIFIED_PATH = '/app/data/stratified'
            cls.KNOWLEDGE_BASE = '/app/data/knowledge_base.csv'
            cls.PATH_TEMP = '/app/temp_files'
            logger.info("Initialized Docker paths")
        else:
            logger.info(f"Using configured paths: ROOT_PATH={cls.ROOT_PATH}")

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Initialize paths before validation
Config.initialize_paths()
Config.validate_paths()