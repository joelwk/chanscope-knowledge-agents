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
        original_len = len(key)

        pass

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

    # Data paths - always relative to project root
    ROOT_PATH = 'data'
    DATA_PATH = 'data'
    ALL_DATA = 'data/all_data.csv'
    ALL_DATA_STRATIFIED_PATH = 'data/stratified'
    KNOWLEDGE_BASE = 'data/knowledge_base.csv'
    PATH_TEMP = 'temp_files'

    # Load and validate OpenAI key first as it's required
    _raw_openai_key = os.getenv('OPENAI_API_KEY')
    logger.info(
        f"Raw OPENAI_API_KEY loaded: {'Yes' if _raw_openai_key else 'No'}")
    if _raw_openai_key:
        logger.debug(
            f"Raw key format: {_raw_openai_key[:5]}...{_raw_openai_key[-5:]}")

    # OpenAI settings with validation
    OPENAI_API_KEY = _clean_api_key(_raw_openai_key)
    if OPENAI_API_KEY:
        logger.info(
            f"Cleaned OPENAI_API_KEY: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-5:]}"
        )
    else:
        logger.error("OPENAI_API_KEY is not set or was cleaned to empty")

    # Flask settings
    FLASK_APP = 'api.app'
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')

    # API settings
    DEFAULT_BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    DEFAULT_SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE'))
    DEFAULT_MAX_WORKERS = int(os.getenv('MAX_WORKERS'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'

    # Model settings
    DEFAULT_EMBEDDING_PROVIDER = os.getenv('DEFAULT_EMBEDDING_PROVIDER','openai')
    DEFAULT_CHUNK_PROVIDER = os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai')
    DEFAULT_SUMMARY_PROVIDER = os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai')

    # OpenAI settings with model validation
    OPENAI_MODEL = _validate_model_name(os.getenv('OPENAI_MODEL', 'gpt-4o'),'openai', 'completion')
    OPENAI_EMBEDDING_MODEL = _validate_model_name(os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),'openai', 'embedding')

    # Grok settings with model validation
    GROK_API_KEY = _clean_api_key(os.getenv('GROK_API_KEY'))
    GROK_MODEL = _validate_model_name(os.getenv('GROK_MODEL', 'grok-2-1212'), 'grok', 'completion')
    GROK_EMBEDDING_MODEL = _validate_model_name(os.getenv('GROK_EMBEDDING_MODEL', 'grok-v1-embedding'), 'grok','embedding')

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
    SELECT_BOARD = _parse_none_value(os.getenv('SELECT_BOARD'))  # Now properly handles "None"
    PADDING_ENABLED = os.getenv('PADDING_ENABLED', 'false')
    CONTRACTION_MAPPING_ENABLED = os.getenv('CONTRACTION_MAPPING_ENABLED','false')
    NON_ALPHA_NUMERIC_ENABLED = os.getenv('NON_ALPHA_NUMERIC_ENABLED', 'false')

    # API Configuration
    API_HOST = os.getenv('API_HOST')  # Host for
    # Use Replit's PORT env var if available, otherwise use API_PORT from env or default to 5000
    API_PORT = os.getenv('API_PORT')
    API_TIMEOUT = 1500  # 25 minutes in seconds

    @classmethod
    def _get_base_urls(cls):
        """Get the base URLs with proper fallback for Replit environment."""
        # Check if we're in Replit
        repl_slug = os.getenv('REPL_SLUG')
        repl_owner = os.getenv('REPL_OWNER')
        repl_port = os.getenv('PORT')  # Replit's assigned port

        urls = []

        # Add Replit URL if in Replit environment
        if repl_slug and repl_owner:
            urls.append(f"https://{repl_slug}.{repl_owner}.repl.co")

        # Add local URL with appropriate port
        # In Replit, use their assigned port
        port = repl_port or cls.API_PORT
        urls.append(f"http://{cls.API_HOST}:{port}")

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
            "embedding_provider": cls.DEFAULT_EMBEDDING_PROVIDER,
            "chunk_provider": cls.DEFAULT_CHUNK_PROVIDER,
            "summary_provider": cls.DEFAULT_SUMMARY_PROVIDER,
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
            "batch_size": cls.DEFAULT_BATCH_SIZE,
            "sample_size": cls.DEFAULT_SAMPLE_SIZE,
            "max_workers": cls.DEFAULT_MAX_WORKERS,
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

# Validate paths on import
Config.validate_paths()