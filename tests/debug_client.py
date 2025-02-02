import json
import pytest
import os
import gc
from pathlib import Path
import logging
from config.settings import Config
from knowledge_agents.model_ops import ModelProvider
import pytest_asyncio
import asyncio
from quart import Quart

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DebugClient:
    def __init__(self, test_client=None):
        """Initialize debug client with Quart test client."""
        self.test_client = test_client

    async def _make_request(self, method, endpoint, **kwargs):
        """Make request using Quart test client."""
        try:
            if method == 'GET':
                response = await self.test_client.get(endpoint)
            elif method == 'POST':
                response = await self.test_client.post(endpoint, json=kwargs.get('json'))
            else:
                raise ValueError(f"Unsupported method: {method}")

            data = await response.get_json()
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content: {data}")
            return data
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {str(e)}")
            return None

    async def test_health(self):
        """Test the API health endpoint."""
        return await self._make_request('GET', '/health')

    async def run_single_query(self, query_data):
        """Run a single query through the API."""
        return await self._make_request('POST', '/process_query', json=query_data)

    async def run_batch_query(self, batch_data):
        """Run a batch of queries through the API."""
        return await self._make_request('POST', '/batch_process', json=batch_data)

    async def check_provider_health(self, provider):
        """Check health of a specific provider."""
        return await self._make_request('GET', f'/health/provider/{provider}')

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="module")
async def app():
    """Create the Quart app for testing."""
    from api import create_app
    app = create_app()
    return app

@pytest_asyncio.fixture
async def test_client(app):
    """Create a test client."""
    return app.test_client()

@pytest_asyncio.fixture
async def debug_client(test_client):
    """Create debug client fixture."""
    return DebugClient(test_client=test_client)

class TestDebugClient:
    """Test debug client functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self, debug_client):
        """Test health check endpoint."""
        response = await debug_client.test_health()
        assert response is not None
        assert response["status"] == "healthy"
        assert response["message"] == "Service is running"
        assert "environment" in response

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", [
        provider.value for provider in ModelProvider
    ])
    async def test_provider_health(self, debug_client, provider):
        """Test provider health checks."""
        response = await debug_client.check_provider_health(provider)
        assert response is not None
        assert "status" in response
        assert response["status"] == "connected"
        assert "provider" in response
        assert response["provider"] == provider
        assert isinstance(response["latency_ms"], (int, float))

    @pytest.mark.asyncio
    async def test_invalid_provider(self, debug_client):
        """Test health check with invalid provider."""
        response = await debug_client.check_provider_health("invalid_provider")
        assert response is not None
        assert "error" in response
        assert "Invalid provider" in response["error"]

    @pytest.mark.asyncio
    async def test_single_query(self, debug_client):
        """Test single query processing."""
        # Get settings from Config
        processing_settings = Config.get_processing_settings()
        model_settings = Config.get_model_settings()
        sample_settings = Config.get_sample_settings()
        
        query_data = {
            "query": "Short test query",
            "batch_size": processing_settings['batch_size'],
            "max_workers": processing_settings['max_workers'],
            "force_refresh": False,
            "sample_size": sample_settings['min_sample_size'],  # Use minimum for tests
            "embedding_provider": model_settings['default_embedding_provider'],
            "chunk_provider": model_settings['default_chunk_provider'],
            "summary_provider": model_settings['default_summary_provider'],
            "cache_enabled": processing_settings['cache_enabled']
        }
        response = await debug_client.run_single_query(query_data)
        assert response is not None
        assert "success" in response
        assert response["success"] is True
        assert "results" in response

    @pytest.mark.asyncio
    async def test_batch_query(self, debug_client):
        """Test batch query processing."""
        # Get settings from Config
        processing_settings = Config.get_processing_settings()
        model_settings = Config.get_model_settings()
        sample_settings = Config.get_sample_settings()
        
        batch_data = {
            "queries": [
                "First test query",
                "Second test query"
            ],
            "batch_size": processing_settings['batch_size'],
            "max_workers": processing_settings['max_workers'],
            "force_refresh": False,
            "sample_size": sample_settings['min_sample_size'],  # Use minimum for tests
            "embedding_provider": model_settings['default_embedding_provider'],
            "chunk_provider": model_settings['default_chunk_provider'],
            "summary_provider": model_settings['default_summary_provider'],
            "cache_enabled": processing_settings['cache_enabled']
        }
        response = await debug_client.run_batch_query(batch_data)
        assert response is not None
        assert "success" in response
        assert response["success"] is True
        assert "results" in response
        assert len(response["results"]) == 2

def load_test_queries(file_path):
    """Load test queries from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load test queries: {e}")
        return None

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 