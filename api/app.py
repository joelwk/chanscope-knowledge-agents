import os
import logging
from flask import Flask
from config.settings import Config
from knowledge_agents import KnowledgeAgentConfig
from .routes import register_routes

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def create_app():
    # Create Flask app
    app = Flask(__name__)

    # Load configuration
    config = Config()
    logger.info("Flask app created and configured with environment:")
    logger.info(f"- FLASK_ENV: {os.getenv('FLASK_ENV', 'development')}")
    logger.info(f"- API_PORT: {os.getenv('API_PORT', '5000')}")
    logger.info(f"- DOCKER_ENV: {os.getenv('DOCKER_ENV', 'false')}")

    # Create knowledge agent configuration
    knowledge_config = KnowledgeAgentConfig.from_env(os.environ)
    app.config['KNOWLEDGE_CONFIG'] = knowledge_config

    # Register routes
    register_routes(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.getenv('API_PORT', 5000))) 