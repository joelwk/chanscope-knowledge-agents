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

if __name__ == '__main__':
    app = create_app()
    config = Config()
    
    # Set minimal configuration needed
    app.config['TIMEOUT'] = 300  # 5 minutes timeout
    app.config['KNOWLEDGE_CONFIG'] = {
        'DEFAULT_SAMPLE_SIZE': config.DEFAULT_SAMPLE_SIZE,
        'DEFAULT_BATCH_SIZE': config.DEFAULT_BATCH_SIZE,
        'DEFAULT_MAX_WORKERS': config.DEFAULT_MAX_WORKERS,
        'PROVIDERS': config.get_provider_settings()
    }

    if is_replit_env():
        port = 5000
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