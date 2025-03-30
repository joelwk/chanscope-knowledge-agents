#!/usr/bin/env python
"""
ChanScopeDataManager Integration Tests

This module provides integration tests for the ChanScopeDataManager class,
using mocked storage implementations to test the Chanscope approach in both
Docker and Replit environments.
"""

import os
import pytest
import logging
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import the required modules
from config.chanscope_config import ChanScopeConfig
from config.env_loader import detect_environment
from knowledge_agents.data_processing.chanscope_manager import ChanScopeDataManager
from config.storage import (
    CompleteDataStorage, StratifiedSampleStorage, EmbeddingStorage, StateManager
)

# Sample data for testing
SAMPLE_DF = pd.DataFrame({
    'thread_id': ['thread1', 'thread2', 'thread3'],
    'posted_date_time': [
        '2023-01-01T00:00:00Z', 
        '2023-01-02T00:00:00Z', 
        '2023-01-03T00:00:00Z'
    ],
    'text': ['Sample text 1', 'Sample text 2', 'Sample text 3']
})

SAMPLE_EMBEDDINGS = np.random.rand(3, 128)  # 3 samples, 128-dimensional embeddings
SAMPLE_THREAD_MAP = {
    'thread1': 0,
    'thread2': 1,
    'thread3': 2
}

class MockCompleteDataStorage(AsyncMock, CompleteDataStorage):
    """Mock implementation of CompleteDataStorage for testing."""
    
    async def store_data(self, df):
        return True
    
    async def get_data(self, filter_date=None):
        return SAMPLE_DF
    
    async def is_data_fresh(self):
        return True
    
    async def get_row_count(self):
        return len(SAMPLE_DF)

class MockStratifiedSampleStorage(AsyncMock, StratifiedSampleStorage):
    """Mock implementation of StratifiedSampleStorage for testing."""
    
    def __init__(self, sample_exists=True):
        super().__init__()
        self._sample_exists = sample_exists
    
    async def store_sample(self, df):
        return True
    
    async def get_sample(self):
        if self._sample_exists:
            return SAMPLE_DF
        return None
    
    async def sample_exists(self):
        return self._sample_exists

class MockEmbeddingStorage(AsyncMock, EmbeddingStorage):
    """Mock implementation of EmbeddingStorage for testing."""
    
    def __init__(self, embeddings_exist=True):
        super().__init__()
        self._embeddings_exist = embeddings_exist
    
    async def store_embeddings(self, embeddings, thread_id_map):
        return True
    
    async def get_embeddings(self):
        if self._embeddings_exist:
            return SAMPLE_EMBEDDINGS, SAMPLE_THREAD_MAP
        return None, None
    
    async def embeddings_exist(self):
        return self._embeddings_exist

class MockStateManager(AsyncMock, StateManager):
    """Mock implementation of StateManager for testing."""
    
    def __init__(self):
        super().__init__()
        self.state = {}
        self.operations = {}
    
    async def update_state(self, state):
        self.state.update(state)
    
    async def get_state(self):
        return self.state
    
    async def mark_operation_start(self, operation):
        self.operations[operation] = {
            'status': 'running',
            'start_time': datetime.now().isoformat()
        }
    
    async def mark_operation_complete(self, operation, result=None):
        if operation in self.operations:
            self.operations[operation].update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'result': result
            })
    
    async def is_operation_in_progress(self, operation):
        return operation in self.operations and self.operations[operation].get('status') == 'running'

@pytest.fixture
def config():
    """Create a basic configuration for testing."""
    return ChanScopeConfig(
        root_data_path=Path("data"),
        stratified_data_path=Path("data/stratified"),
        temp_path=Path("temp"),
        sample_size=100,
        time_column='posted_date_time'
    )

@pytest.fixture
def mock_storage_empty():
    """Create mock storage implementations with no existing data."""
    return {
        'complete_data': MockCompleteDataStorage(),
        'stratified_sample': MockStratifiedSampleStorage(sample_exists=False),
        'embeddings': MockEmbeddingStorage(embeddings_exist=False),
        'state': MockStateManager()
    }

@pytest.fixture
def mock_storage_with_data():
    """Create mock storage implementations with existing data."""
    return {
        'complete_data': MockCompleteDataStorage(),
        'stratified_sample': MockStratifiedSampleStorage(sample_exists=True),
        'embeddings': MockEmbeddingStorage(embeddings_exist=True),
        'state': MockStateManager()
    }

@pytest.mark.asyncio
async def test_ensure_data_ready_force_refresh_empty(config, mock_storage_empty):
    """
    Test ensure_data_ready with force_refresh=True when no data exists.
    
    This tests the Chanscope approach logic for forced refresh:
    - Should check if complete data exists and is fresh
    - Should create a new stratified sample
    - Should generate new embeddings
    """
    # Extract storage implementations
    complete_data_storage = mock_storage_empty['complete_data']
    stratified_storage = mock_storage_empty['stratified_sample']
    embedding_storage = mock_storage_empty['embeddings']
    state_manager = mock_storage_empty['state']
    
    # Create data manager with mocked storage
    manager = ChanScopeDataManager(
        config=config,
        complete_data_storage=complete_data_storage,
        stratified_storage=stratified_storage,
        embedding_storage=embedding_storage,
        state_manager=state_manager
    )
    
    # Run the test
    result = await manager.ensure_data_ready(force_refresh=True)
    
    # Verify data was loaded correctly
    assert result is True
    
    # Verify complete data was loaded
    complete_data_storage.get_row_count.assert_called()
    complete_data_storage.is_data_fresh.assert_called()
    
    # Verify stratified sample was created (force_refresh=True)
    stratified_storage.store_sample.assert_called()
    
    # Verify embeddings were generated (force_refresh=True)
    embedding_storage.store_embeddings.assert_called()
    
    # Verify state manager was updated
    state_manager.mark_operation_start.assert_called_with("ensure_data_ready")
    state_manager.mark_operation_complete.assert_called()

@pytest.mark.asyncio
async def test_ensure_data_ready_force_refresh_false_with_data(config, mock_storage_with_data):
    """
    Test ensure_data_ready with force_refresh=False when data exists.
    
    This tests the Chanscope approach logic for non-forced refresh:
    - Should use existing stratified sample
    - Should use existing embeddings
    """
    # Extract storage implementations
    complete_data_storage = mock_storage_with_data['complete_data']
    stratified_storage = mock_storage_with_data['stratified_sample']
    embedding_storage = mock_storage_with_data['embeddings']
    state_manager = mock_storage_with_data['state']
    
    # Create data manager with mocked storage
    manager = ChanScopeDataManager(
        config=config,
        complete_data_storage=complete_data_storage,
        stratified_storage=stratified_storage,
        embedding_storage=embedding_storage,
        state_manager=state_manager
    )
    
    # Run the test
    result = await manager.ensure_data_ready(force_refresh=False)
    
    # Verify result
    assert result is True
    
    # Verify checks were made for existing data
    complete_data_storage.get_row_count.assert_called()
    stratified_storage.get_sample.assert_called()
    embedding_storage.embeddings_exist.assert_called()
    
    # Verify no new stratified sample was created (using existing)
    stratified_storage.store_sample.assert_not_called()
    
    # Verify no new embeddings were generated (using existing)
    embedding_storage.store_embeddings.assert_not_called()
    
    # Verify state manager was updated
    state_manager.mark_operation_start.assert_called_with("ensure_data_ready")
    state_manager.mark_operation_complete.assert_called()

@pytest.mark.asyncio
async def test_ensure_data_ready_force_refresh_false_no_data(config, mock_storage_empty):
    """
    Test ensure_data_ready with force_refresh=False when no data exists.
    
    This tests the Chanscope approach logic for non-forced refresh with missing data:
    - Should create new stratified sample when none exists
    - Should generate new embeddings when none exist
    """
    # Extract storage implementations
    complete_data_storage = mock_storage_empty['complete_data']
    stratified_storage = mock_storage_empty['stratified_sample']
    embedding_storage = mock_storage_empty['embeddings']
    state_manager = mock_storage_empty['state']
    
    # Create data manager with mocked storage
    manager = ChanScopeDataManager(
        config=config,
        complete_data_storage=complete_data_storage,
        stratified_storage=stratified_storage,
        embedding_storage=embedding_storage,
        state_manager=state_manager
    )
    
    # Run the test
    result = await manager.ensure_data_ready(force_refresh=False)
    
    # Verify result
    assert result is True
    
    # Verify checks were made for existing data
    complete_data_storage.get_row_count.assert_called()
    stratified_storage.get_sample.assert_called()
    embedding_storage.embeddings_exist.assert_called()
    
    # Verify new stratified sample was created (none exists)
    stratified_storage.store_sample.assert_called()
    
    # Verify new embeddings were generated (none exist)
    embedding_storage.store_embeddings.assert_called()
    
    # Verify state manager was updated
    state_manager.mark_operation_start.assert_called_with("ensure_data_ready")
    state_manager.mark_operation_complete.assert_called()

@pytest.mark.asyncio
async def test_factory_method():
    """Test the factory method creates a ChanScopeDataManager with the correct storage implementations."""
    with patch('knowledge_agents.data_processing.chanscope_manager.StorageFactory') as mock_factory:
        # Create mock storage
        mock_storage = {
            'complete_data': MockCompleteDataStorage(),
            'stratified_sample': MockStratifiedSampleStorage(),
            'embeddings': MockEmbeddingStorage(),
            'state': MockStateManager()
        }
        mock_factory.create.return_value = mock_storage
        
        # Create config
        config = ChanScopeConfig(
            root_data_path=Path("data"),
            stratified_data_path=Path("data/stratified"),
            temp_path=Path("temp")
        )
        
        # Create data manager using factory method
        manager = ChanScopeDataManager.create_for_environment(config)
        
        # Verify factory was called
        mock_factory.create.assert_called_once_with(config, getattr(config, 'env', None))
        
        # Verify storage implementations were set correctly
        assert manager.complete_data_storage == mock_storage['complete_data']
        assert manager.stratified_storage == mock_storage['stratified_sample']
        assert manager.embedding_storage == mock_storage['embeddings']
        assert manager.state_manager == mock_storage['state'] 