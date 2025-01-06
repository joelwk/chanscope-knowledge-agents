import os
from dotenv import load_dotenv
from pathlib import Path

# Find and load environment variables from root .env file
root_dir = Path(__file__).parent.parent.resolve()
env_path = root_dir / ".env"

if not env_path.exists():
    raise FileNotFoundError(
        f"Root .env file not found at {env_path}. "
        "Please ensure it exists and you're running from the project root."
    )

load_dotenv(env_path)

class Config:
    """Base configuration."""
    # Project root
    PROJECT_ROOT = root_dir
    
    # Flask settings
    FLASK_APP = 'api.app'
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    
    # Data paths - resolve relative to project root if not absolute
    ROOT_PATH = os.getenv('ROOT_PATH', str(root_dir / 'data'))
    DATA_PATH = os.getenv('DATA_PATH', str(root_dir / 'data'))
    ALL_DATA = os.getenv('ALL_DATA', str(root_dir / 'data/all_data.csv'))
    ALL_DATA_STRATIFIED_PATH = os.getenv('ALL_DATA_STRATIFIED_PATH', str(root_dir / 'data/stratified'))
    KNOWLEDGE_BASE = os.getenv('KNOWLEDGE_BASE', str(root_dir / 'data/knowledge_base.csv'))
    PATH_TEMP = os.getenv('PATH_TEMP', str(root_dir / 'temp_files'))
    
    # API settings
    DEFAULT_BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
    DEFAULT_MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    
    # Model settings
    DEFAULT_EMBEDDING_PROVIDER = os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai')
    DEFAULT_CHUNK_PROVIDER = os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai')
    DEFAULT_SUMMARY_PROVIDER = os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai')

    @classmethod
    def validate_paths(cls):
        """Ensure all required paths exist"""
        paths = [
            Path(cls.ROOT_PATH),
            Path(cls.DATA_PATH),
            Path(cls.ALL_DATA_STRATIFIED_PATH),
            Path(cls.PATH_TEMP)
        ]
        
        for path in paths:
            # If path is relative, make it absolute relative to project root
            if not path.is_absolute():
                path = Path(cls.PROJECT_ROOT) / path
            path.mkdir(parents=True, exist_ok=True)

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