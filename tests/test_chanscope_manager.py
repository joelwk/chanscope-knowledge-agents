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

class MockCompleteDataStorage:
    """Mock implementation of CompleteDataStorage for testing."""

    def __init__(self):
        self.store_data = AsyncMock(return_value=True)
        self.get_data = AsyncMock(return_value=SAMPLE_DF)
        self.is_data_fresh = AsyncMock(return_value=True)
        self.get_row_count = AsyncMock(return_value=len(SAMPLE_DF))

class MockStratifiedSampleStorage:
    """Mock implementation of StratifiedSampleStorage for testing."""

    def __init__(self, sample_exists=True):
        self._sample_exists = sample_exists
        self.store_sample = AsyncMock(side_effect=self._store_sample)
        self.get_sample = AsyncMock(side_effect=self._get_sample)
        self.sample_exists = AsyncMock(side_effect=self._sample_exists_fn)

    async def _get_sample(self):
        if self._sample_exists:
            return SAMPLE_DF
        return None

    async def _store_sample(self, df):
        self._sample_exists = True
        return True

    async def _sample_exists_fn(self):
        return self._sample_exists

class MockEmbeddingStorage:
    """Mock implementation of EmbeddingStorage for testing."""

    def __init__(self, embeddings_exist=True):
        self._embeddings_exist = embeddings_exist
        self.store_embeddings = AsyncMock(side_effect=self._store_embeddings)
        self.get_embeddings = AsyncMock(side_effect=self._get_embeddings)
        self.embeddings_exist = AsyncMock(side_effect=self._embeddings_exist_fn)

    async def _get_embeddings(self):
        if self._embeddings_exist:
            return SAMPLE_EMBEDDINGS, SAMPLE_THREAD_MAP
        return None, None

    async def _store_embeddings(self, embeddings, thread_id_map):
        self._embeddings_exist = True
        return True

    async def _embeddings_exist_fn(self):
        return self._embeddings_exist

class MockStateManager:
    """Mock implementation of StateManager for testing."""

    def __init__(self):
        self.state = {}
        self.operations = {}
        self.update_state = AsyncMock(side_effect=self._update_state)
        self.get_state = AsyncMock(side_effect=self._get_state)
        self.mark_operation_start = AsyncMock(side_effect=self._mark_operation_start)
        self.mark_operation_complete = AsyncMock(side_effect=self._mark_operation_complete)
        self.is_operation_in_progress = AsyncMock(side_effect=self._is_operation_in_progress)

    async def _update_state(self, state):
        self.state.update(state)

    async def _get_state(self):
        return self.state

    async def _mark_operation_start(self, operation):
        self.operations[operation] = {
            'status': 'running',
            'start_time': datetime.now().isoformat()
        }

    async def _mark_operation_complete(self, operation, result=None):
        if operation in self.operations:
            self.operations[operation].update({
                'status': 'completed',
                'end_time': datetime.now().isoformat(),
                'result': result
            })

    async def _is_operation_in_progress(self, operation):
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
    state_manager.mark_operation_start.assert_any_call("ensure_data_ready")
    state_manager.mark_operation_complete.assert_any_call("ensure_data_ready", "success")

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
    state_manager.mark_operation_start.assert_any_call("ensure_data_ready")
    state_manager.mark_operation_complete.assert_any_call("ensure_data_ready", "success")

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
    state_manager.mark_operation_start.assert_any_call("ensure_data_ready")
    state_manager.mark_operation_complete.assert_any_call("ensure_data_ready", "success")

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
