"""Configuration settings module."""
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from config.base_settings import get_base_settings, ensure_base_paths
import copy

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages configuration settings for the application."""

    def __init__(self, base_settings: Optional[Dict[str, Any]] = None):
        """Initialize configuration manager with base settings."""
        if base_settings is None:
            base_settings = get_base_settings()
        self._settings = copy.deepcopy(base_settings)
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate settings structure and types."""
        required_categories = ['processing', 'model', 'sample', 'columns', 'paths']
        for category in required_categories:
            if category not in self._settings:
                raise ValueError(f"Missing required category: {category}")

    def get_model_settings(self) -> Dict[str, Any]:
        """Get model-related settings."""
        return self._settings.get('model', {})

    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing-related settings."""
        return self._settings.get('processing', {})

    def get_sample_settings(self) -> Dict[str, Any]:
        """Get sample-related settings."""
        return self._settings.get('sample', {})

    def get_column_settings(self) -> Dict[str, Any]:
        """Get column-related settings."""
        return self._settings.get('columns', {})

    def get_path_settings(self) -> Dict[str, Any]:
        """Get path-related settings."""
        return self._settings.get('paths', {})

    def get_api_settings(self) -> Dict[str, Any]:
        """Get API-related settings."""
        return self._settings.get('api', {})

    def get_aws_settings(self) -> Dict[str, Any]:
        """Get AWS-related settings."""
        return self._settings.get('aws', {})

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._settings

# Initialize configuration manager with base settings
config_manager = ConfigurationManager()

@lru_cache()
def get_settings() -> Dict[str, Any]:
    """Get all settings."""
    base_settings = get_base_settings()
    paths = base_settings['paths']
    
    return {
        # Path Settings
        'BASE_DIR': paths['root'],
        'LOGS_DIR': paths['logs'],
        'TEMP_DIR': paths['temp'],
        
        # Server Settings
        'HOST': base_settings['api']['host'],
        'PORT': base_settings['api']['port'],
        'WORKERS': base_settings['processing']['max_workers'],
        
        # Database Settings
        'DATABASE_URL': base_settings['api'].get('database_url', ''),
        'ASYNC_DATABASE_URL': base_settings['api'].get('async_database_url'),
        
        # Cache Settings
        'REDIS_URL': base_settings['api'].get('redis_url', 'redis://localhost:6379'),
        'CACHE_TTL': base_settings['processing'].get('cache_ttl', 3600),
        'ENABLE_REDIS': base_settings['processing'].get('cache_enabled', False),
        
        # Model Settings
        'MODEL_NAME': base_settings['model']['chunk_model'],
        'EMBEDDING_MODEL': base_settings['model']['embedding_model'],
        
        # Security Settings
        'SECRET_KEY': base_settings['api'].get('secret_key', ''),
        'API_KEY': base_settings['api'].get('api_key', ''),
        'ALLOWED_HOSTS': base_settings['api'].get('allowed_hosts', '*').split(','),
        
        # Logging Settings
        'LOG_LEVEL': base_settings['api'].get('log_level', 'INFO'),
        'LOG_FORMAT': base_settings['api'].get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    }

# Global settings instance
settings = get_settings()

# Ensure required directories exist
ensure_base_paths()

class Config:
    """Configuration class for accessing settings."""
    
    @classmethod
    def get_model_settings(cls) -> Dict[str, Any]:
        """Get model-related settings."""
        return config_manager.get_model_settings()
    
    @classmethod
    def get_data_settings(cls) -> Dict[str, Any]:
        """Get data-related settings."""
        return config_manager.get_all_settings().get('data', {})
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings."""
        return config_manager.get_all_settings()
        
    @classmethod
    def get_api_settings(cls) -> Dict[str, Any]:
        """Get API-related settings."""
        return config_manager.get_api_settings()
        
    @classmethod
    def get_paths(cls) -> Dict[str, Any]:
        """Get path-related settings."""
        return config_manager.get_path_settings()
        
    @classmethod
    def get_processing_settings(cls) -> Dict[str, Any]:
        """Get processing-related settings."""
        return config_manager.get_processing_settings()
        
    @classmethod
    def get_sample_settings(cls) -> Dict[str, Any]:
        """Get sample-related settings."""
        return config_manager.get_sample_settings()

    @classmethod
    def get_column_settings(cls) -> Dict[str, Any]:
        """Get column-related settings."""
        return config_manager.get_column_settings()

    @classmethod
    def get_chunk_settings(cls) -> Dict[str, Any]:
        """Get chunk-related settings."""
        processing_settings = config_manager.get_processing_settings()
        return {
            'processing_chunk_size': processing_settings.get('processing_chunk_size', 5000),
            'stratification_chunk_size': processing_settings.get('stratification_chunk_size', 10000)
        }

    @classmethod
    def validate_settings(cls) -> Dict[str, bool]:
        """Validate all required settings are present and properly formatted."""
        validation = {}
        
        # Validate paths
        paths = cls.get_paths()
        validation['paths'] = all(key in paths for key in ['root_data_path', 'stratified', 'temp'])
        
        # Validate column settings
        column_settings = cls.get_column_settings()
        validation['columns'] = all(key in column_settings for key in ['time_column', 'strata_column', 'column_types'])
        
        # Validate model settings
        model_settings = cls.get_model_settings()
        validation['model'] = all(key in model_settings for key in [
            'default_embedding_provider',
            'default_chunk_provider',
            'default_summary_provider',
            'embedding_batch_size',
            'process_chunk_batch_size',
            'chunk_batch_size',
            'summary_batch_size'
        ])
        
        # Validate sample settings
        sample_settings = cls.get_sample_settings()
        validation['sample'] = all(key in sample_settings for key in [
            'max_sample_size',
            'default_sample_size',
            'min_sample_size'
        ])
        
        return validation

    @classmethod
    def get_openai_api_key(cls) -> Optional[str]:
        """Get OpenAI API key from settings."""
        api_settings = cls.get_api_settings()
        return api_settings.get('openai_api_key')

    @classmethod
    def get_grok_api_key(cls) -> Optional[str]:
        """Get Grok API key from settings."""
        api_settings = cls.get_api_settings()
        return api_settings.get('grok_api_key')

    @classmethod
    def get_venice_api_key(cls) -> Optional[str]:
        """Get Venice API key from settings."""
        api_settings = cls.get_api_settings()
        return api_settings.get('venice_api_key')

    @classmethod
    def get_openai_model(cls) -> Optional[str]:
        """Get OpenAI model from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('chunk_model')

    @classmethod
    def get_openai_embedding_model(cls) -> str:
        """Get OpenAI embedding model from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('embedding_model', 'text-embedding-3-large')

    @classmethod
    def get_grok_model(cls) -> Optional[str]:
        """Get Grok model from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('grok_model')

    @classmethod
    def get_venice_model(cls) -> Optional[str]:
        """Get Venice model from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('venice_model')

    @classmethod
    def get_venice_chunk_model(cls) -> Optional[str]:
        """Get Venice chunk model from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('venice_chunk_model')

    @classmethod
    def get_default_embedding_provider(cls) -> str:
        """Get default embedding provider from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('default_embedding_provider', 'openai')

    @classmethod
    def get_default_chunk_provider(cls) -> str:
        """Get default chunk provider from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('default_chunk_provider', 'venice')

    @classmethod
    def get_default_summary_provider(cls) -> str:
        """Get default summary provider from settings."""
        model_settings = cls.get_model_settings()
        return model_settings.get('default_summary_provider', 'venice')

    @classmethod
    def get_aws_settings(cls) -> Dict[str, Any]:
        """Get AWS settings from configuration."""
        aws_settings = config_manager.get_aws_settings()
        # Rename keys to match expected format
        return {
            'aws_access_key_id': aws_settings['access_key_id'],
            'aws_secret_access_key': aws_settings['secret_access_key'],
            'aws_default_region': aws_settings['region'],
            's3_bucket': aws_settings['s3_bucket'],
            's3_bucket_prefix': aws_settings['s3_bucket_prefix'],
            's3_bucket_processed': aws_settings['s3_bucket_processed'],
            's3_bucket_models': aws_settings['s3_bucket_models']
        }

# Initialize paths
ensure_base_paths()

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