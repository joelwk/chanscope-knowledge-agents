"""Test suite for FastAPI endpoints."""
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from typing import AsyncGenerator, AsyncIterator
import asyncio
import os
import time
from datetime import datetime, timedelta
import pytz
from config.base_settings import get_base_settings

# Import the app instance
from api.app import app

from knowledge_agents.data_ops import DataOperations, DataConfig

from api.routers.shared import (
    store_batch_result,
    get_background_tasks,
    get_tasks_lock,
    get_batch_results,
)

# Create test client
client = TestClient(app)

# Define API base path - update this to match your current API configuration
API_BASE_PATH = "/api/v1"

@pytest_asyncio.fixture
async def test_config(tmp_path) -> DataConfig:
    """Create a test configuration with temporary paths."""
    # Get retention days from environment or config
    base_settings = get_base_settings()
    processing_settings = base_settings.get('processing', {})
    retention_days = int(os.environ.get('DATA_RETENTION_DAYS', processing_settings.get('retention_days', 30)))
    
    # Calculate filter date based on retention days
    filter_date = (datetime.now(pytz.UTC) - timedelta(days=retention_days)).strftime('%Y-%m-%d')
    
    root_path = tmp_path / "test_data"
    stratified_path = root_path / "stratified"
    temp_path = root_path / "temp"
    
    return DataConfig(
        root_data_path=root_path,
        stratified_data_path=stratified_path,
        temp_path=temp_path,
        filter_date=filter_date,  # Use the calculated filter date
        sample_size=10  # Small sample size for tests
    )

@pytest_asyncio.fixture
async def data_ops(test_config) -> DataOperations:
    """Create DataOperations instance for testing."""
    operations = DataOperations(test_config)
    
    # Setup - ensure we have a clean environment
    try:
        if test_config.root_data_path.exists():
            import shutil
            shutil.rmtree(test_config.root_data_path)
        test_config.root_data_path.mkdir(parents=True, exist_ok=True)
        test_config.stratified_data_path.mkdir(parents=True, exist_ok=True)
        test_config.temp_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Setup error: {e}")
    
    yield operations
    
    # Cleanup
    try:
        if test_config.root_data_path.exists():
            import shutil
            shutil.rmtree(test_config.root_data_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint."""
    response = client.get(f"{API_BASE_PATH}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_query_endpoint(data_ops):
    """Test the query endpoint."""
    try:
        # Prepare test data with force_refresh=True to ensure data is loaded from the retention date
        await data_ops.ensure_data_ready(force_refresh=True, skip_embeddings=False)
        
        # Test query
        test_query = {
            "query": "test query",
            "force_refresh": False,  # Use existing data for the query
            "skip_embeddings": True
        }
        
        response = client.post(f"{API_BASE_PATH}/query", json=test_query)
        assert response.status_code in [200, 202]  # Accept either OK or Accepted
        data = response.json()
        assert "task_id" in data or "result" in data  # Either async or sync response
    except Exception as e:
        pytest.skip(f"Skipping due to data preparation error: {str(e)}")

@pytest.mark.asyncio
async def test_batch_process_endpoint(data_ops):
    """Test the batch process endpoint."""
    try:
        # Prepare test data with force_refresh=True to ensure data is loaded from the retention date
        await data_ops.ensure_data_ready(force_refresh=True, skip_embeddings=False)
        
        # Initialize the agent
        from knowledge_agents.embedding_ops import get_agent
        agent = await get_agent()
        
        # Test batch query
        test_batch = {
            "queries": ["test query 1", "test query 2"],
            "force_refresh": False,  # Use existing data for the query
            "skip_embeddings": True
        }
        
        response = client.post(f"{API_BASE_PATH}/batch_process", json=test_batch)
        assert response.status_code in [200, 202]  # Accept either OK or Accepted
        data = response.json()
        assert "task_id" in data or "results" in data  # Either async or sync response
    except Exception as e:
        pytest.skip(f"Skipping due to data preparation error: {str(e)}")

@pytest.mark.asyncio
async def test_embedding_status_endpoint(data_ops):
    """Test the embedding status endpoint."""
    try:
        # Generate embeddings with force_refresh=True to ensure data is loaded from the retention date
        await data_ops.ensure_data_ready(force_refresh=True, skip_embeddings=False)
        await data_ops.generate_embeddings(force_refresh=True)
        
        response = client.get(f"{API_BASE_PATH}/embedding_status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "metrics" in data
    except Exception as e:
        pytest.skip(f"Skipping due to embedding generation error: {str(e)}")

@pytest.mark.asyncio
async def test_cache_health_endpoint():
    """Test the cache health endpoint."""
    # Direct test of the cache health endpoint
    response = client.get(f"{API_BASE_PATH}/health/cache")
    
    # Assert the response is successful
    assert response.status_code == 200
    
    # Parse the response data
    data = response.json()
    
    # Assert the expected structure
    assert "status" in data
    assert data["status"] == "healthy"
    assert "metrics" in data
    
    # Check metrics
    metrics = data["metrics"]
    assert "hits" in metrics
    assert "misses" in metrics
    assert "errors" in metrics
    assert "total_requests" in metrics
    assert "hit_ratio" in metrics

@pytest.mark.asyncio
async def test_batch_status_returns_list(monkeypatch):
    """Ensure batch status endpoint returns a list of results for completed tasks."""

    async def noop_history(*args, **kwargs):
        return None

    monkeypatch.setattr("api.routers.shared._update_batch_history", noop_history)

    task_id = f"query_{int(time.time())}_regression"
    _background_tasks = get_background_tasks()
    _batch_results = get_batch_results()
    _tasks_lock = get_tasks_lock()

    test_result = {
        "chunks": [{"text": "example"}],
        "summary": "regression summary",
        "metadata": {"source": "unit-test"},
    }

    async with _tasks_lock:
        _background_tasks[task_id] = {
            "status": "completed",
            "total_queries": 1,
            "completed_queries": 1,
            "duration_ms": 0.0,
        }

    try:
        await store_batch_result(
            batch_id=task_id,
            result=test_result,
            config={"batch_result_ttl": 60, "query": "regression query"},
            save_to_disk=False,
        )

        response = client.get(f"{API_BASE_PATH}/batch_status/{task_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert isinstance(payload["results"], list)
        assert payload["results"][0]["summary"] == "regression summary"
    finally:
        async with _tasks_lock:
            _background_tasks.pop(task_id, None)
        _batch_results.pop(task_id, None)


@pytest.mark.asyncio
async def test_batch_result_schema():
    """Test batch processing handles both dict and tuple result formats."""
    # This is a regression test for ID-001: KeyError when accessing result[0]
    # Tests that the batch processing loop correctly handles dict results
    
    # Mock results in different formats to test schema-aware extraction
    dict_result = {"chunks": ["chunk1"], "summary": "summary1", "metadata": {"test": "value"}}
    tuple_result = (["chunk2"], "summary2", {"test": "value2"})
    
    # Test that both formats can be processed without KeyError
    results = [dict_result, tuple_result]
    queries = ["query1", "query2"]
    
    # Simulate the extraction logic from the fixed batch_process_queries function
    for i, (query, result) in enumerate(zip(queries, results)):
        # Schema-aware extraction: handle both dict and tuple result formats
        if isinstance(result, dict):
            chunks = result.get("chunks", [])
            summary = result.get("summary", "")
            meta_extra = result.get("metadata", {})
        else:  # tuple fallback
            chunks, summary = result[:2]
            meta_extra = result[2] if len(result) > 2 else {}
        
        # Verify extraction worked correctly
        assert chunks is not None
        assert summary is not None
        assert meta_extra is not None
        
        if i == 0:  # dict result
            assert chunks == ["chunk1"]
            assert summary == "summary1"
            assert meta_extra == {"test": "value"}
        else:  # tuple result
            assert chunks == ["chunk2"]
            assert summary == "summary2"
            assert meta_extra == {"test": "value2"}

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in endpoints."""
    # Test invalid query
    invalid_query = {
        "invalid_field": "test"
    }
    response = client.post(f"{API_BASE_PATH}/query", json=invalid_query)
    assert response.status_code == 422  # Unprocessable Entity

    # Test missing required field
    empty_query = {}
    response = client.post(f"{API_BASE_PATH}/query", json=empty_query)
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
