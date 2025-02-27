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
    response = await client.get('/health_replit')
    assert response.status_code == 200
    data = await response.get_json()
    assert 'status' in data or 'message' in data

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
    data = await client.get_json()
    assert 'status' in data
    assert 'provider' in data
    assert 'latency_ms' in data

@pytest.mark.asyncio
async def test_health_connections(client):
    """Test the service connections endpoint."""
    response = await client.get('/health/connections')
    assert response.status_code == 200
    data = await client.get_json()
    assert 'services' in data
    assert bool(data['services'])

@pytest.mark.asyncio
async def test_health_all(client):
    """Test the all providers health endpoint."""
    response = await client.get('/health/all')
    assert response.status_code == 200
    data = await client.get_json()
    assert isinstance(data, dict)
    assert len(data) > 0

@pytest.mark.asyncio
async def test_process_recent_query(client):
    """Test the recent query processing endpoint."""
    response = await client.get('/process_recent_query')
    assert response.status_code == 200
    data = await client.get_json()
    for key in ['query', 'time_range', 'chunks', 'summary']:
        assert key in data

@pytest.mark.asyncio
async def test_process_query(client):
    """Test the process query endpoint."""
    payload = {
         "query": "test query",
         "force_refresh": False,
         "skip_embeddings": True,
         "filter_date": None
    }
    response = await client.post('/process_query', json=payload)
    assert response.status_code == 200
    data = await response.get_json()
    assert 'results' in data
    for key in ['query', 'chunks', 'summary']:
        assert key in data['results']

@pytest.mark.asyncio
async def test_batch_process(client):
    """Test the batch process endpoint."""
    payload = {
         "queries": ["test query 1", "test query 2"],
         "force_refresh": False,
         "embedding_batch_size": 50,
         "chunk_batch_size": 5000,
         "summary_batch_size": 50,
         "embedding_provider": "openai",
         "chunk_provider": "openai",
         "summary_provider": "openai"
    }
    response = await client.post('/batch_process', json=payload)
    assert response.status_code == 200
    data = await response.get_json()
    assert 'results' in data
    assert isinstance(data['results'], list)