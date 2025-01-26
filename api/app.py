import os
import logging
from config.settings import Config
from . import create_app, is_replit_env

# Setup logging only if handlers don't exist
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

def is_docker_env():
    """Check if running in Docker environment."""
    return os.getenv('DOCKER_ENV') == 'true'

# Create the app instance at module level
app = create_app()

if __name__ == '__main__':
    config = Config()
    port = os.getenv('PORT', 5000)  # Get port from environment or default to 5000
    
    if is_docker_env():
        logger.info("Starting app in Docker environment")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    elif is_replit_env():
        logger.info(f"Starting app in Replit environment: host=0.0.0.0, port={port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    else:
        logger.info(f"Starting app locally: host={config.API_HOST}, port={config.API_PORT}")
        app.run(
            host=config.API_HOST,
            port=config.API_PORT,
            debug=True
        )