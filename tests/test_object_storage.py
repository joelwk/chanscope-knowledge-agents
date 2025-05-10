"""
Tests for object storage integration with query results.
"""

import os
import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.asyncio

from knowledge_agents.utils import save_query_output, _save_to_object_storage, _save_to_filesystem


# Mock data for testing
SAMPLE_RESPONSE = {
    "status": "completed",
    "task_id": "test_task_id",
    "query": "test query",
    "chunks": [
        {"thread_id": "123", "content": "Test content 1"},
        {"thread_id": "456", "content": "Test content 2"}
    ],
    "summary": "Test summary",
    "metadata": {"test_key": "test_value"}
}


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock()


@pytest.fixture
def mock_client():
    """Create a mock Replit Object Storage client."""
    mock = MagicMock()
    mock.upload_from_text.return_value = None
    mock.upload_from_bytes.return_value = None
    return mock


@patch('knowledge_agents.config.env_loader.is_replit_environment')
@patch('replit.object_storage.Client')
async def test_save_to_object_storage(mock_client_class, mock_is_replit, mock_logger):
    """Test saving query results to object storage."""
    # Setup
    mock_is_replit.return_value = True
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    
    # Test function
    json_key, embeddings_key = await _save_to_object_storage(
        SAMPLE_RESPONSE,
        {},  # Empty embeddings data
        "test_prefix",
        True,
        True,
        mock_logger
    )
    
    # Assertions
    assert json_key is not None
    assert "test_prefix" in json_key
    assert mock_client_instance.upload_from_text.call_count == 1
    
    # Check the content that was uploaded
    uploaded_content = mock_client_instance.upload_from_text.call_args[0][1]
    parsed_content = json.loads(uploaded_content)
    assert parsed_content["task_id"] == "test_task_id"
    assert parsed_content["query"] == "test query"


@patch('knowledge_agents.config.env_loader.is_replit_environment')
async def test_save_query_output_replit_env(mock_is_replit, mock_logger, tmp_path):
    """Test save_query_output in Replit environment."""
    # Setup
    mock_is_replit.return_value = True
    
    # Mock the _save_to_object_storage function
    with patch('knowledge_agents.utils._save_to_object_storage') as mock_save_to_object_storage:
        mock_save_to_object_storage.return_value = ("test_json_key", "test_embeddings_key")
        
        # Test function
        json_path, embeddings_path = await save_query_output(
            response=SAMPLE_RESPONSE,
            base_path=tmp_path,
            logger=mock_logger
        )
        
        # Assertions
        assert json_path == "test_json_key"
        assert embeddings_path == "test_embeddings_key"
        mock_save_to_object_storage.assert_called_once()


@patch('knowledge_agents.config.env_loader.is_replit_environment')
async def test_save_query_output_docker_env(mock_is_replit, mock_logger, tmp_path):
    """Test save_query_output in Docker environment."""
    # Setup
    mock_is_replit.return_value = False
    
    # Create test directories
    (tmp_path / "embeddings").mkdir(exist_ok=True)
    
    # Test function
    json_path, embeddings_path = await save_query_output(
        response=SAMPLE_RESPONSE,
        base_path=tmp_path,
        logger=mock_logger
    )
    
    # Assertions
    assert json_path is not None
    assert json_path.parent == tmp_path
    assert json_path.exists()
    
    # Check file contents
    with open(json_path, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data["task_id"] == "test_task_id"
    assert saved_data["query"] == "test query"


@patch('knowledge_agents.config.env_loader.is_replit_environment')
async def test_save_query_output_fallback(mock_is_replit, mock_logger, tmp_path):
    """Test fallback to filesystem when object storage fails."""
    # Setup
    mock_is_replit.return_value = True
    (tmp_path / "embeddings").mkdir(exist_ok=True)
    
    # Mock the _save_to_object_storage function to simulate a failure
    with patch('knowledge_agents.utils._save_to_object_storage') as mock_save_to_object_storage:
        mock_save_to_object_storage.side_effect = Exception("Object storage failure")
        
        # Test function
        json_path, embeddings_path = await save_query_output(
            response=SAMPLE_RESPONSE,
            base_path=tmp_path,
            logger=mock_logger
        )
        
        # Assertions
        assert json_path is not None
        assert json_path.parent == tmp_path
        assert json_path.exists()
        
        # Verify filesystem fallback was used
        assert isinstance(json_path, Path)  # Object storage returns strings, filesystem returns Path objects 