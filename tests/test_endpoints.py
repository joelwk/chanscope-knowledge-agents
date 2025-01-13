import pytest
import os
from api import create_app
import pytest_asyncio

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="module")
async def app():
    """Create a test client for the app."""
    return create_app()

@pytest_asyncio.fixture
async def client(app, event_loop):
    """Create an async test client."""
    async with app.test_client() as client:
        yield client

@pytest.mark.asyncio
async def test_health_check(client):
    """Test the health check endpoint."""
    response = await client.get('/health')
    assert response.status_code == 200
    data = await response.get_json()
    assert data['status'] == 'healthy'
    assert 'message' in data

@pytest.mark.asyncio
async def test_replit_health(client):
    """Test the health check endpoint in Replit environment."""
    os.environ['QUART_ENV'] = 'replit'
    response = await client.get('/health')
    assert response.status_code == 200
    data = await response.get_json()
    assert data['status'] == 'healthy'
    assert 'message' in data

@pytest.mark.asyncio
async def test_s3_health(client):
    """Test the S3 health check endpoint."""
    response = await client.get('/health/s3')
    assert response.status_code == 200
    data = await response.get_json()
    assert 's3_status' in data
    assert 'bucket_access' in data
    assert 'latency_ms' in data

@pytest.mark.asyncio
async def test_provider_health(client):
    """Test the provider health check endpoint."""
    response = await client.get('/health/provider/openai')
    assert response.status_code == 200
    data = await response.get_json()
    assert 'status' in data
    assert 'provider' in data
    assert 'latency_ms' in data