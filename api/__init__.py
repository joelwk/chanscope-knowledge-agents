from quart import Quart
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
    config = Config()
    logger.info("=== App Configuration ===")
    logger.info(f"- QUART_ENV: {os.getenv('QUART_ENV', 'development')}")
    logger.info(f"- API_HOST: {config.API_HOST}")
    logger.info(f"- API_PORT: {config.API_PORT}")
    logger.info(f"- REPLIT_ENV: {is_replit_env()}")

    # Log all possible base URLs
    api_settings = config.get_api_settings()
    base_urls = api_settings.get('base_urls', [])
    logger.info("Available API endpoints:")
    for url in base_urls:
        logger.info(f"- {url}")

    if is_replit_env():
        logger.info("=== Replit Environment Details ===")
        logger.info(f"- REPL_SLUG: {os.getenv('REPL_SLUG')}")
        logger.info(f"- REPL_OWNER: {os.getenv('REPL_OWNER')}")
        logger.info(f"- REPL_PORT: {os.getenv('PORT')}")
        logger.info(
            f"- Expected URL: https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co"
        )

    # Create knowledge agent configuration
    app.config['KNOWLEDGE_CONFIG'] = {
        'PATHS': Config.get_data_paths(),
        'ROOT_PATH': Config.ROOT_PATH,
        'PROVIDERS': Config.get_provider_settings(),
        'SAMPLE_SIZE': Config.SAMPLE_SIZE,
        'MAX_WORKERS': Config.MAX_WORKERS,
        'CACHE_ENABLED': Config.CACHE_ENABLED
    }
    
    # Import routes
    from .routes import register_routes
    register_routes(app)
    
    return app 