"""Base settings configuration without dependencies."""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pytz

# Global settings cache
_settings_cache = None
_env_loaded = False

def get_base_paths() -> Dict[str, Path]:
    """Get base paths without any dependencies."""
    root = Path(__file__).parent.parent.resolve()
    return {
        'root': root,
        'config': root / 'config',
        'root_data_path': root / 'data',
        'stratified': root / 'data/stratified',
        'generated_data': root / 'data/generated_data',
        'embeddings': root / 'data/generated_data/embeddings',
        'temp': root / 'temp_files',
        'logs': root / 'logs'
    }
    
def ensure_base_paths() -> None:
    """Ensure base paths exist."""
    paths = get_base_paths()
    for name, path in paths.items():
        # Create directories for all paths
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured path exists: {path}")

def get_env_path() -> Path:
    """Get the path to the .env file."""
    return get_base_paths()['root'] / '.env'

def load_env_vars() -> None:
    """Load environment variables."""
    global _env_loaded
    
    # Skip if environment is already loaded
    if _env_loaded or os.getenv('ENVIRONMENT_LOADED'):
        logging.debug("Environment variables already loaded, skipping")
        return
        
    env_path = get_env_path()
    if env_path.exists():
        load_dotenv(env_path)
        _env_loaded = True
        os.environ['ENVIRONMENT_LOADED'] = 'true'
        logging.info("Environment variables loaded from .env file")

def get_base_settings() -> Dict[str, Any]:
    """Get base settings from environment variables.
    
    This is the single source of truth for all environment variables.
    All other modules should import and use these settings rather than
    reading environment variables directly.
    """
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    # Load environment variables if not already loaded
    load_env_vars()
    
    # Format the filter date in ISO format for consistent parsing
    try:
        retention_days = int(os.getenv('DATA_RETENTION_DAYS', '30'))
    except (ValueError, TypeError):
        logging.warning("Invalid DATA_RETENTION_DAYS value, using default of 30 days")
        retention_days = 30
        
    filter_date = (
        datetime.now(pytz.UTC) - timedelta(days=retention_days)
    ).isoformat()
    
    _settings_cache = {
        'api': {
            'host': os.getenv('API_HOST', '0.0.0.0').strip(),
            'port': int(os.getenv('API_PORT', '80').strip()),
            'base_path': os.getenv('API_BASE_PATH', '/api/v1').strip(),
            'base_url': os.getenv('API_BASE_URL', '').strip(),
            'docker_env': os.getenv('DOCKER_ENV', 'false').lower().strip() == 'true',
            'openai_api_key': os.getenv('OPENAI_API_KEY', '').strip(),
            'grok_api_key': os.getenv('GROK_API_KEY', '').strip(),
            'venice_api_key': os.getenv('VENICE_API_KEY', '').strip(),
            'use_mock_data': os.getenv('USE_MOCK_DATA', 'false').lower().strip() in ('true', 'yes', '1'),
            'use_mock_embeddings': os.getenv('USE_MOCK_EMBEDDINGS', 'false').lower().strip() in ('true', 'yes', '1')
        },
        'model': {
            'default_embedding_provider': os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai').strip(),
            'default_chunk_provider': os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai').strip(),
            'default_summary_provider': os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai').strip(),
            'embedding_batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '50').strip()),
            'chunk_batch_size': int(os.getenv('CHUNK_BATCH_SIZE', '5000').strip()),
            'summary_batch_size': int(os.getenv('SUMMARY_BATCH_SIZE', '50').strip()),
            'embedding_model': os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large').strip(),
            'chunk_model': os.getenv('OPENAI_MODEL', '').strip(),
            'summary_model': os.getenv('OPENAI_MODEL', '').strip(),
            'grok_model': os.getenv('GROK_MODEL', '').strip(),
            'venice_model': os.getenv('VENICE_MODEL', '').strip(),
            'venice_chunk_model': os.getenv('VENICE_CHUNK_MODEL', '').strip(),
            'venice_character_slug': os.getenv('VENICE_CHARACTER_SLUG', 'venice').strip(),
            'openai_api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1').strip(),
            'grok_api_base': os.getenv('GROK_API_BASE', '').strip(),
            'venice_api_base': os.getenv('VENICE_API_BASE', '').strip(),
            'openai_chunk_model': os.getenv('OPENAI_CHUNK_MODEL', '').strip(),
            'grok_chunk_model': os.getenv('GROK_CHUNK_MODEL', '').strip()
        },
        'aws': {
            'access_key_id': os.getenv('AWS_ACCESS_KEY_ID', '').strip(),
            'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '').strip(),
            'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1').strip(),
            's3_bucket': os.getenv('S3_BUCKET', 'chanscope-data').strip(),
            's3_bucket_prefix': os.getenv('S3_BUCKET_PREFIX', 'data/').strip(),
            's3_bucket_processed': os.getenv('S3_BUCKET_PROCESSED', 'processed').strip(),
            's3_bucket_models': os.getenv('S3_BUCKET_MODELS', 'models').strip()
        },
        'processing': {
            'processing_chunk_size': int(os.getenv('PROCESSING_CHUNK_SIZE', '10000')),
            'stratification_chunk_size': int(os.getenv('STRATIFICATION_CHUNK_SIZE', '5000')),
            'padding_enabled': os.getenv('PADDING_ENABLED', 'false').lower() == 'true',
            'contraction_mapping_enabled': os.getenv('CONTRACTION_MAPPING_ENABLED', 'false').lower() == 'true',
            'non_alpha_numeric_enabled': os.getenv('NON_ALPHA_NUMERIC_ENABLED', 'false').lower() == 'true',
            'max_tokens': int(os.getenv('MAX_TOKENS', '4096')),
            'max_workers': int(os.getenv('MAX_WORKERS')),
            'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            'retention_days': retention_days,
            'filter_date': os.getenv('FILTER_DATE', filter_date),
            'select_board': os.getenv('SELECT_BOARD'),
            'use_batching': os.getenv('USE_BATCHING', 'true').lower() == 'true',
            'cache_ttl': int(os.getenv('CACHE_TTL', '3600')),
            'batch_size': int(os.getenv('BATCH_SIZE', '64')),
            'use_mock_data': os.getenv('USE_MOCK_DATA', 'false').lower() == 'true',
            'use_mock_embeddings': os.getenv('USE_MOCK_EMBEDDINGS', 'false').lower() == 'true'
        },
        'sample': {
            'max_sample_size': 100000,
            'default_sample_size': int(os.getenv('SAMPLE_SIZE', '1500')),
            'min_sample_size': 100
        },
        'columns': {
            'time_column': os.getenv('TIME_COLUMN', 'posted_date_time'),
            'strata_column': os.getenv('STRATA_COLUMN', None),
            'column_types': {
                'thread_id': str,
                'posted_date_time': str,
                'text_clean': str,
                'posted_comment': str
            }
        },
        'paths': get_base_paths()
    }
    
    return _settings_cache

# Initialize environment variables and cache settings on module import
load_env_vars() 