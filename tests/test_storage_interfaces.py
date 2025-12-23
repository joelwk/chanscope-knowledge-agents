#!/usr/bin/env python
"""
Storage Interface Tests

This module provides unit tests for the storage interfaces used by
Chanscope. It tests both file-based (Docker) and database (Replit)
storage implementations.
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
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
from config.storage import (
    CompleteDataStorage, StratifiedSampleStorage, EmbeddingStorage, StateManager,
    FileCompleteDataStorage, FileStratifiedSampleStorage, FileEmbeddingStorage, FileStateManager,
    ReplitCompleteDataStorage, ReplitStratifiedSampleStorage, ReplitEmbeddingStorage, ReplitStateManager,
    StorageFactory
)
from config.chanscope_config import ChanScopeConfig
from config.env_loader import detect_environment

# Test data
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

SAMPLE_STATE = {
    'last_update': datetime.now().isoformat(),
    'status': 'completed',
    'records_processed': 100
}

@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary configuration for testing file-based storage."""
    # Create test directories
    root_path = tmp_path / "data"
    root_path.mkdir()
    strat_path = root_path / "stratified"
    strat_path.mkdir()
    temp_path = tmp_path / "temp"
    temp_path.mkdir()
    
    # Create config
    config = ChanScopeConfig(
        root_data_path=root_path,
        stratified_data_path=strat_path,
        temp_path=temp_path,
        env='docker'  # Force Docker environment
    )
    
    return config

@pytest.fixture
def mock_replit_db():
    """Create a mock replit db for testing."""
    mock_db = {}
    
    class MockDB(dict):
        def __setitem__(self, key, value):
            mock_db[key] = value
        
        def __getitem__(self, key):
            return mock_db.get(key, None)
        
        def __delitem__(self, key):
            if key in mock_db:
                del mock_db[key]
        
        def __contains__(self, key):
            return key in mock_db
        
        def prefix(self, prefix):
            return [k for k in mock_db.keys() if k.startswith(prefix)]
    
    return MockDB()

@pytest.fixture
def replit_config(tmp_path):
    """Create a configuration for testing Replit storage."""
    # Create test directories
    root_path = tmp_path / "data"
    root_path.mkdir()
    strat_path = root_path / "stratified"
    strat_path.mkdir()
    temp_path = tmp_path / "temp"
    temp_path.mkdir()
    
    # Create config
    config = ChanScopeConfig(
        root_data_path=root_path,
        stratified_data_path=strat_path,
        temp_path=temp_path,
        env='replit'  # Force Replit environment
    )
    
    return config

#
# File Storage Tests
#

@pytest.mark.asyncio
async def test_file_complete_data_storage(temp_config):
    """Test file-based complete data storage."""
    storage = FileCompleteDataStorage(temp_config)
    
    # Test storing data
    result = await storage.store_data(SAMPLE_DF)
    assert result is True
    
    # Test file exists
    complete_data_path = temp_config.root_data_path / "complete_data.csv"
    assert complete_data_path.exists()
    
    # Test retrieving data
    df = await storage.get_data()
    assert not df.empty
    assert len(df) == len(SAMPLE_DF)
    
    # Test data freshness
    is_fresh = await storage.is_data_fresh()
    assert is_fresh is True
    
    # Test row count
    row_count = await storage.get_row_count()
    assert row_count == len(SAMPLE_DF)

@pytest.mark.asyncio
async def test_file_stratified_sample_storage(temp_config):
    """Test file-based stratified sample storage."""
    storage = FileStratifiedSampleStorage(temp_config)
    
    # Test storing sample
    result = await storage.store_sample(SAMPLE_DF)
    assert result is True
    
    # Test file exists
    stratified_path = temp_config.stratified_data_path / "stratified_sample.csv"
    assert stratified_path.exists()
    
    # Test sample existence
    exists = await storage.sample_exists()
    assert exists is True
    
    # Test retrieving sample
    df = await storage.get_sample()
    assert not df.empty
    assert len(df) == len(SAMPLE_DF)

@pytest.mark.asyncio
async def test_file_embedding_storage(temp_config):
    """Test file-based embedding storage."""
    storage = FileEmbeddingStorage(temp_config)
    
    # Test storing embeddings
    result = await storage.store_embeddings(SAMPLE_EMBEDDINGS, SAMPLE_THREAD_MAP)
    assert result is True
    
    # Test files exist
    embeddings_path = temp_config.stratified_data_path / "embeddings.npz"
    thread_map_path = temp_config.stratified_data_path / "thread_id_map.json"
    assert embeddings_path.exists()
    assert thread_map_path.exists()
    
    # Test embeddings existence
    exists = await storage.embeddings_exist()
    assert exists is True
    
    # Test retrieving embeddings
    embeddings, thread_map = await storage.get_embeddings()
    assert embeddings is not None
    assert thread_map is not None
    assert embeddings.shape == SAMPLE_EMBEDDINGS.shape
    assert set(thread_map.keys()) == set(SAMPLE_THREAD_MAP.keys())

@pytest.mark.asyncio
async def test_file_state_manager(temp_config):
    """Test file-based state manager."""
    storage = FileStateManager(temp_config)
    
    # Test updating state
    await storage.update_state(SAMPLE_STATE)
    
    # Test getting state
    state = await storage.get_state()
    assert state is not None
    assert state.get('status') == SAMPLE_STATE['status']
    
    # Test marking operation start
    await storage.mark_operation_start("test_operation")
    
    # Test operation in progress
    in_progress = await storage.is_operation_in_progress("test_operation")
    assert in_progress is True
    
    # Test marking operation complete
    await storage.mark_operation_complete("test_operation", {"result": "success"})
    
    # Test operation no longer in progress
    in_progress = await storage.is_operation_in_progress("test_operation")
    assert in_progress is False

#
# Replit Storage Tests (with mocks)
#

@pytest.mark.asyncio
async def test_replit_complete_data_storage(replit_config):
    """Test Replit-based complete data storage."""
    # Mock the PostgresDB class
    with patch('config.storage.PostgresDB') as mock_postgres:
        mock_instance = MagicMock()
        mock_postgres.return_value = mock_instance
        
        # Mock database methods expected by storage
        mock_instance.sync_data_from_dataframe.return_value = len(SAMPLE_DF)
        mock_instance.get_complete_data.return_value = SAMPLE_DF
        mock_instance.check_data_needs_update.return_value = (False, None)
        mock_instance.get_row_count.return_value = len(SAMPLE_DF)
        
        storage = ReplitCompleteDataStorage(replit_config)
        
        # Test storing data
        result = await storage.store_data(SAMPLE_DF)
        assert result is True
        
        # Test retrieving data
        df = await storage.get_data()
        assert not df.empty
        
        # Test data freshness
        is_fresh = await storage.is_data_fresh()
        assert is_fresh is True
        
        # Test row count
        row_count = await storage.get_row_count()
        assert row_count > 0

@pytest.mark.asyncio
async def test_replit_stratified_sample_storage(replit_config, mock_replit_db):
    """Test Replit-based stratified sample storage."""
    with patch('config.storage.db', mock_replit_db):
        storage = ReplitStratifiedSampleStorage(replit_config)
        
        # Test storing sample
        result = await storage.store_sample(SAMPLE_DF)
        assert result is True
        
        # Test sample existence
        exists = await storage.sample_exists()
        assert exists is True
        
        # Test retrieving sample
        df = await storage.get_sample()
        assert df is not None

@pytest.mark.asyncio
async def test_replit_embedding_storage(replit_config, mock_replit_db):
    """Test Replit-based embedding storage."""
    with patch('config.storage.db', mock_replit_db):
        storage = ReplitEmbeddingStorage(replit_config)
        
        # Test storing embeddings
        result = await storage.store_embeddings(SAMPLE_EMBEDDINGS, SAMPLE_THREAD_MAP)
        assert result is True
        
        # Test embeddings existence
        exists = await storage.embeddings_exist()
        assert exists is True
        
        # Test retrieving embeddings
        embeddings, thread_map = await storage.get_embeddings()
        assert embeddings is not None
        assert thread_map is not None

@pytest.mark.asyncio
async def test_replit_state_manager(replit_config, mock_replit_db):
    """Test Replit-based state manager."""
    with patch('config.storage.db', mock_replit_db):
        storage = ReplitStateManager(replit_config)
        
        # Test updating state
        await storage.update_state(SAMPLE_STATE)
        
        # Test getting state
        state = await storage.get_state()
        assert state is not None
        
        # Test marking operation start
        await storage.mark_operation_start("test_operation")
        
        # Test operation in progress
        in_progress = await storage.is_operation_in_progress("test_operation")
        assert in_progress is True
        
        # Test marking operation complete
        await storage.mark_operation_complete("test_operation", {"result": "success"})
        
        # Test operation no longer in progress
        in_progress = await storage.is_operation_in_progress("test_operation")
        assert in_progress is False

@pytest.mark.asyncio
async def test_storage_factory():
    """Test the StorageFactory creates appropriate implementations."""
    # Test Docker environment
    docker_config = ChanScopeConfig(
        root_data_path=Path("data"),
        stratified_data_path=Path("data/stratified"),
        temp_path=Path("temp"),
        env='docker'
    )
    docker_storage = StorageFactory.create(docker_config)
    
    assert isinstance(docker_storage['complete_data'], FileCompleteDataStorage)
    assert isinstance(docker_storage['stratified_sample'], FileStratifiedSampleStorage)
    assert isinstance(docker_storage['embeddings'], FileEmbeddingStorage)
    assert isinstance(docker_storage['state'], FileStateManager)
    
    # Test Replit environment
    replit_config = ChanScopeConfig(
        root_data_path=Path("data"),
        stratified_data_path=Path("data/stratified"),
        temp_path=Path("temp"),
        env='replit'
    )
    replit_storage = StorageFactory.create(replit_config)
    
    assert isinstance(replit_storage['complete_data'], ReplitCompleteDataStorage)
    assert isinstance(replit_storage['stratified_sample'], ReplitStratifiedSampleStorage)
    assert isinstance(replit_storage['embeddings'], ReplitEmbeddingStorage)
    assert isinstance(replit_storage['state'], ReplitStateManager) 
