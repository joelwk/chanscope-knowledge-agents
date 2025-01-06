from flask import Flask
from flask_cors import CORS
import logging
from dotenv import load_dotenv
import os
from pathlib import Path

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

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Configure logging
    if not app.debug:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)
    
    # Import routes
    from .routes import register_routes
    register_routes(app)
    
    # Log that the app was created with environment info
    logger.info("Flask app created and configured with environment:")
    logger.info(f"- FLASK_ENV: {os.getenv('FLASK_ENV', 'development')}")
    logger.info(f"- API_PORT: {os.getenv('API_PORT', 5000)}")
    logger.info(f"- DOCKER_ENV: {os.getenv('DOCKER_ENV', 'false')}")
    
    return app 