from quart import Quart, jsonify
from quart_cors import cors
import logging
from dotenv import load_dotenv
import os
from pathlib import Path
from config.settings import Config

logger = logging.getLogger(__name__)

# Setup logging
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load environment variables from root .env
root_dir = Path(__file__).parent.parent.resolve()
env_path = root_dir / ".env"

if not env_path.exists():
    logger.error(f"Root .env file not found at {env_path}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Directory contents: {list(root_dir.iterdir())}")
    raise FileNotFoundError(
        f"Root .env file not found at {env_path}. "
        "Please ensure it exists and you're running the app from the correct directory."
    )

logger.info(f"Loading environment from root: {env_path}")
load_dotenv(env_path)

def is_replit_env():
    """Check if running in Replit environment."""
    return bool(os.getenv('REPL_SLUG')) and bool(os.getenv('REPL_OWNER'))

def create_app():
    """Create and configure the Quart application."""
    app = Quart(__name__)
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    # Enable CORS for all routes
    app = cors(app)
    
    # Configure logging
    if not app.debug:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)
    else:
        # Disable default logging in debug mode
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.ERROR)

    # Load configuration
    api_settings = Config.get_api_settings()
    logger.info("=== App Configuration ===")
    logger.info(f"- QUART_ENV: {api_settings['quart_env']}")
    logger.info(f"- API_HOST: {api_settings['host']}")
    logger.info(f"- API_PORT: {api_settings['port']}")
    logger.info(f"- REPLIT_ENV: {is_replit_env()}")

    # Log available API endpoints
    logger.info("Available API endpoints:")
    logger.info(f"- http://{api_settings['host']}:{api_settings['port']}")

    if is_replit_env():
        logger.info("=== Replit Environment Details ===")
        logger.info(f"- REPL_SLUG: {os.getenv('REPL_SLUG')}")
        logger.info(f"- REPL_OWNER: {os.getenv('REPL_OWNER')}")
        logger.info(f"- REPL_PORT: {os.getenv('PORT')}")
        logger.info(
            f"- Expected URL: https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co"
        )

    # Create knowledge agent configuration
    paths = Config.get_paths()
    model_settings = Config.get_model_settings()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    
    app.config['KNOWLEDGE_CONFIG'] = {
        'PATHS': paths,
        'ROOT_PATH': paths['root_data_path'],
        'PROVIDERS': {
            'embedding_provider': model_settings['default_embedding_provider'],
            'chunk_provider': model_settings['default_chunk_provider'],
            'summary_provider': model_settings['default_summary_provider']
        },
        'SAMPLE_SIZE': sample_settings['default_sample_size'],
        'MAX_WORKERS': processing_settings['max_workers'],
        'CACHE_ENABLED': processing_settings['cache_enabled']
    }
    
    # Import routes
    from .routes import bp, get_api_docs
    
    # In Replit environment, mount all routes under /api
    if is_replit_env():
        # Register root route at / that shows the API documentation
        @app.route('/')
        async def root():
            return jsonify(get_api_docs("/api"))
            
        # Register blueprint with /api prefix
        app.register_blueprint(bp, url_prefix='/api')
    else:
        app.register_blueprint(bp)
    
    return app 