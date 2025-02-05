"""Configuration settings module."""
import os
import logging
from pathlib import Path
from config.base import ROOT_DIR, get_base_paths, ensure_base_paths
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class Config:
    """Base configuration."""

    # Load base paths
    base_paths = get_base_paths()
    PROJECT_ROOT = base_paths['root']

    # Environment detection
    @classmethod
    def is_docker_env(cls) -> bool:
        """Detect if running in Docker environment."""
        # Check for common Docker environment indicators
        docker_indicators = [
            Path('/.dockerenv').exists(),  # Standard Docker environment file
            Path('/run/.containerenv').exists(),  # Podman/container environment file
            any('docker' in line.lower() for line in Path('/proc/1/cgroup').read_text().splitlines()) if Path('/proc/1/cgroup').exists() else False
        ]
        return any(docker_indicators)

    @classmethod
    def get_environment_base_path(cls) -> str:
        """Get the base path for the current environment."""
        if cls.is_docker_env():
            return '/app'
        return str(cls.PROJECT_ROOT)

    # Centralized Configuration Constants
    CHUNK_SIZE_SETTINGS = {
        'min_chunk_size': int(os.getenv('CHUNK_SIZE')),
        'max_chunk_size': 5000,  # Safety limit
        'default_chunk_size': int(os.getenv('CHUNK_SIZE')),
        'processing_chunk_size': int(os.getenv('CHUNK_BATCH_SIZE')),
        'stratification_chunk_size': int(os.getenv('SUMMARY_BATCH_SIZE'))
    }

    COLUMN_DEFINITIONS = {
        'time_column': os.getenv('TIME_COLUMN', 'posted_date_time'),
        'strata_column': os.getenv('STRATA_COLUMN', None),
        'required_columns': [
            'thread_id',
            os.getenv('TIME_COLUMN', 'posted_date_time'),
            'text_clean',
            'posted_comment'
        ],
        'column_types': {
            'thread_id': 'object',  # Using pandas object type for strings
            os.getenv('TIME_COLUMN', 'posted_date_time'): 'object',  # Will be converted to datetime separately
            'text_clean': 'object',
            'posted_comment': 'object'
        }
    }

    SAMPLE_SIZE_SETTINGS = {
        'max_sample_size': 100000,  # Safety limit
        'default_sample_size': int(os.getenv('SAMPLE_SIZE')),
        'min_sample_size': 100  # Minimum for statistical validity
    }

    # Model Provider Settings
    MODEL_PROVIDER_SETTINGS = {
        'default_embedding_provider': os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai'),
        'default_chunk_provider': os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai'),
        'default_summary_provider': os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai'),
        'embedding_batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE')),
        'chunk_batch_size': int(os.getenv('CHUNK_BATCH_SIZE')),
        'summary_batch_size': int(os.getenv('SUMMARY_BATCH_SIZE')),
        'embedding_model': os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large'),
        'chunk_model': os.getenv('OPENAI_MODEL', 'gpt-4'),
        'summary_model': os.getenv('OPENAI_MODEL', 'gpt-4'),
        'grok_model': os.getenv('GROK_MODEL', 'grok-4'),
        'venice_model': os.getenv('VENICE_MODEL', 'venice-4'),
        'venice_chunk_model': os.getenv('VENICE_CHUNK_MODEL', 'venice-chunk-4')
    }

    # Processing Settings
    PROCESSING_SETTINGS = {
        'padding_enabled': os.getenv('PADDING_ENABLED', 'false').lower() == 'true',
        'contraction_mapping_enabled': os.getenv('CONTRACTION_MAPPING_ENABLED', 'false').lower() == 'true',
        'non_alpha_numeric_enabled': os.getenv('NON_ALPHA_NUMERIC_ENABLED', 'false').lower() == 'true',
        'max_tokens': int(os.getenv('MAX_TOKENS')),
        'max_workers': int(os.getenv('MAX_WORKERS')),
        'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
        'chunk_size': int(os.getenv('CHUNK_SIZE')),
        'filter_date': None,  # Default to None, allow override from API/UI
        'select_board': os.getenv('SELECT_BOARD')
    }

    # AWS Settings
    AWS_SETTINGS = {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'aws_default_region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        's3_bucket': os.getenv('S3_BUCKET', 'chanscope-data'),
        's3_bucket_prefix': os.getenv('S3_BUCKET_PREFIX', 'data/'),
        's3_bucket_processed': os.getenv('S3_BUCKET_PROCESSED', 'processed'),
        's3_bucket_models': os.getenv('S3_BUCKET_MODELS', 'models')
    }

    # Path Settings
    PATH_SETTINGS = {
        'root_data_path': os.getenv('ROOT_PATH', 'data'),
        'stratified': os.getenv('STRATIFIED_PATH', 'data/stratified'),
        'knowledge_base': os.getenv('KNOWLEDGE_BASE', 'data/knowledge_base.csv'),
        'temp': os.getenv('PATH_TEMP', 'temp_files')
    }

    # API Settings
    API_SETTINGS = {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', 5000)),
        'quart_env': os.getenv('QUART_ENV', 'development'),
        'quart_app': 'api.app',
        'docker_env': os.getenv('DOCKER_ENV', 'false').lower() == 'true',
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'grok_api_key': os.getenv('GROK_API_KEY'),
        'venice_api_key': os.getenv('VENICE_API_KEY'),
        'base_url': os.getenv('API_BASE_URL', 'http://0.0.0.0:5000'),
        'base_urls': [
            'http://0.0.0.0:5000',
            'http://api:5000',
            'http://localhost:5000'
        ]
    }

    @staticmethod
    def _clean_api_key(key: str) -> str:
        """Clean API key by removing angle brackets and quotes."""
        if not key:
            logger.warning("API key is empty or None")
            return key
            
        key = key.strip()
        original_len = len(key)
        
        key = ''.join(c for c in key if c.isprintable()).strip()
        
        if key.startswith("<"):
            key = key[1:]
        if key.endswith(">"):
            key = key[:-1]
        key = key.strip("'\"").strip()
        
        if len(key) != original_len:
            logger.info(f"API key was cleaned (length changed from {original_len} to {len(key)})")
            
        return key

    @staticmethod
    def _parse_none_value(value: str) -> Optional[str]:
        """Parse a string value that might represent None."""
        if not value or value.lower() == 'none':
            return None
        return value.strip()

    @staticmethod
    def _validate_model_name(model_name: str, provider: str, model_type: str) -> str:
        """Validate and clean model name based on provider and type."""
        if not model_name:
            logger.warning(f"Model name for {provider} {model_type} is empty or None")
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
                if not any(model_name.startswith(prefix) for prefix in valid_prefixes[model_type]):
                    logger.warning(f"OpenAI {model_type} model '{model_name}' may be invalid - doesn't start with expected prefix")

        elif provider == 'grok':
            valid_prefixes = {
                'completion': ['grok-'],
                'embedding': ['grok-'],
                'chunk': ['grok-']
            }
            if model_type in valid_prefixes:
                if not any(model_name.startswith(prefix) for prefix in valid_prefixes[model_type]):
                    logger.warning(f"Grok {model_type} model '{model_name}' may be invalid - doesn't start with expected prefix")

        elif provider == 'venice':
            # Venice models have more varied names, just log the model being used
            logger.info(f"Using Venice {model_type} model: {model_name}")

        else:
            logger.warning(f"Unknown provider '{provider}' for model validation")

        return model_name

    @classmethod
    def _parse_date(cls, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date with UTC timezone enforcement."""
        if not date_str:
            return None
        try:
            # Parse with UTC awareness
            date = pd.to_datetime(date_str, utc=True)
            if date.tzinfo is None:
                date = date.tz_localize('UTC')
            return date
        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {e}")
            return None

    @classmethod
    def get_filter_date(cls) -> Optional[datetime]:
        """Single source of truth for filter date.
        
        Checks for runtime override before falling back to environment variable.
        """
        # First check if there's a runtime override in processing settings
        runtime_date = cls.PROCESSING_SETTINGS.get('filter_date')
        if runtime_date is not None:
            return cls._parse_date(runtime_date)
            
        # Fall back to environment variable
        return cls._parse_date(os.getenv('FILTER_DATE'))

    @classmethod
    def get_chunk_settings(cls) -> Dict[str, int]:
        """Get unified chunk size settings."""
        return cls.CHUNK_SIZE_SETTINGS

    @classmethod
    def get_column_settings(cls) -> Dict[str, Any]:
        """Get unified column settings."""
        return cls.COLUMN_DEFINITIONS

    @classmethod
    def get_sample_settings(cls) -> Dict[str, int]:
        """Get unified sample size settings."""
        return cls.SAMPLE_SIZE_SETTINGS

    @classmethod
    def get_model_settings(cls) -> Dict[str, Any]:
        """Get unified model provider settings."""
        return cls.MODEL_PROVIDER_SETTINGS

    @classmethod
    def get_processing_settings(cls) -> Dict[str, Any]:
        """Get unified processing settings."""
        return cls.PROCESSING_SETTINGS

    @classmethod
    def get_aws_settings(cls) -> Dict[str, Any]:
        """Get unified AWS settings."""
        return cls.AWS_SETTINGS

    @classmethod
    def get_paths(cls) -> Dict[str, str]:
        """Get unified path settings with proper validation."""
        if cls.API_SETTINGS['docker_env']:
            # Docker environment paths with absolute paths
            paths = {
                'root_data_path': '/app/data',
                'stratified': '/app/data/stratified',
                'knowledge_base': '/app/data/knowledge_base.csv',
                'temp': '/app/temp_files'
            }
        else:
            # Local environment paths from PATH_SETTINGS
            paths = cls.PATH_SETTINGS.copy()
        
        # Create directories if they don't exist
        for path_key in ['root_data_path', 'stratified', 'temp']:
            path = Path(paths[path_key])
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Error creating directory {path}: {e}")
                raise ValueError(f"Failed to create required directory {path_key}: {e}")
        
        # Create parent directories for file paths
        for path_key in ['knowledge_base']:
            path = Path(paths[path_key])
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created parent directory: {path.parent}")
            except Exception as e:
                logger.error(f"Error creating parent directory for {path}: {e}")
                raise ValueError(f"Failed to create parent directory for {path_key}: {e}")
        
        return paths

    @classmethod
    def get_api_settings(cls) -> Dict[str, Any]:
        """Get unified API settings."""
        api_settings = cls.API_SETTINGS.copy()
        if api_settings['docker_env']:
            api_settings['openai_api_key'] = cls._clean_api_key(api_settings['openai_api_key'])
        return api_settings

    @classmethod
    def validate_paths(cls):
        """Validate and create required paths."""
        paths = cls.get_paths()
        required_paths = ['root_data_path', 'stratified', 'temp']
        missing_paths = []
        
        # Create parent directories for all paths
        for key in required_paths:
            path = Path(paths[key])
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Error creating directory {path}: {e}")
                missing_paths.append(key)
        
        if missing_paths:
            error_msg = f"Missing required paths in configuration: {', '.join(missing_paths)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @classmethod
    def get_path_settings(cls) -> dict:
        """Get path settings."""
        return cls.PATH_SETTINGS

    # API Key getters
    @classmethod
    def get_openai_api_key(cls) -> Optional[str]:
        """Get OpenAI API key."""
        return cls._clean_api_key(cls.API_SETTINGS.get('openai_api_key'))

    @classmethod
    def get_grok_api_key(cls) -> Optional[str]:
        """Get Grok API key."""
        return cls._clean_api_key(cls.API_SETTINGS.get('grok_api_key'))

    @classmethod
    def get_venice_api_key(cls) -> Optional[str]:
        """Get Venice API key."""
        return cls._clean_api_key(cls.API_SETTINGS.get('venice_api_key'))

    # Model getters
    @classmethod
    def get_openai_model(cls) -> str:
        """Get OpenAI model."""
        return cls.MODEL_PROVIDER_SETTINGS.get('chunk_model')

    @classmethod
    def get_openai_embedding_model(cls) -> str:
        """Get OpenAI embedding model."""
        return cls.MODEL_PROVIDER_SETTINGS.get('embedding_model')

    @classmethod
    def get_default_embedding_provider(cls) -> str:
        """Get default embedding provider."""
        return cls.MODEL_PROVIDER_SETTINGS.get('default_embedding_provider')

    @classmethod
    def get_default_chunk_provider(cls) -> str:
        """Get default chunk provider."""
        return cls.MODEL_PROVIDER_SETTINGS.get('default_chunk_provider')

    @classmethod
    def get_default_summary_provider(cls) -> str:
        """Get default summary provider."""
        return cls.MODEL_PROVIDER_SETTINGS.get('default_summary_provider')

    @classmethod
    def get_grok_model(cls) -> str:
        """Get Grok model."""
        return cls.MODEL_PROVIDER_SETTINGS.get('grok_model')

    @classmethod
    def get_venice_model(cls) -> str:
        """Get Venice model."""
        return cls.MODEL_PROVIDER_SETTINGS.get('venice_model')

    @classmethod
    def get_venice_chunk_model(cls) -> str:
        """Get Venice chunk model."""
        return cls.MODEL_PROVIDER_SETTINGS.get('venice_chunk_model')

    @classmethod
    def get_embedding_batch_size(cls) -> int:
        """Get the embedding batch size from model settings."""
        return cls.get_model_settings().get('embedding_batch_size')

    @classmethod
    def get_chunk_batch_size(cls) -> int:
        """Get the chunk batch size from model settings."""
        return cls.get_model_settings().get('chunk_batch_size')

    @classmethod
    def get_summary_batch_size(cls) -> int:
        """Get the summary batch size from model settings."""
        return cls.get_model_settings().get('summary_batch_size')

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
Config.validate_paths()