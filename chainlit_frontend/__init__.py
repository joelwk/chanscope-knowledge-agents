"""
Chainlit Frontend Package
"""
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)
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

# Configure API endpoint
is_docker = os.path.exists('/.dockerenv')  # More reliable Docker detection
api_host = "api" if is_docker else "localhost"
api_port = int(os.getenv('API_PORT', 5000))
API_BASE_URL = f"http://{api_host}:{api_port}"

# Log configuration
logger.info("Chainlit frontend configured with environment:")
logger.info(f"- API endpoint: {API_BASE_URL}")
logger.info(f"- DOCKER_ENV: {os.getenv('DOCKER_ENV')}")
logger.info(f"- SERVICE_TYPE: {os.getenv('SERVICE_TYPE')}")
logger.info(f"- DEFAULT_EMBEDDING_PROVIDER: {os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai')}")
logger.info(f"- DEFAULT_CHUNK_PROVIDER: {os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai')}")
logger.info(f"- DEFAULT_SUMMARY_PROVIDER: {os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai')}") 