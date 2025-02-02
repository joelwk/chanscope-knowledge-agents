"""Test suite for data ingestion functionality."""
import pytest
import asyncio
import pandas as pd
import numpy as np
import json
from pathlib import Path
import shutil
import logging
from datetime import datetime, timedelta
import pytz

from knowledge_agents.data_ops import (
    DataConfig,
    DataOperations,
    DataStateManager,
    prepare_knowledge_base
)
from knowledge_agents.embedding_ops import get_relevant_content
from knowledge_agents.model_ops import ModelProvider
from config.settings import Config

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Test data constants
TEST_DATA = {
    'thread_id': ['1', '2', '3'],
    'posted_date_time': [
        (datetime.now(pytz.UTC) - timedelta(days=i)).isoformat()
        for i in range(3)
    ],
    'text_clean': [
        'Test content 1',
        'Test content 2',
        'Test content 3'
    ],
    'posted_comment': [
        'Comment 1',
        'Comment 2',
        'Comment 3'
    ]
}

@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration with temporary paths."""
    # Get settings from Config
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    
    return DataConfig(
        root_path=tmp_path / "root",
        all_data_path=tmp_path / "data" / "all_data.csv",
        stratified_data_path=tmp_path / "data" / "stratified",
        knowledge_base_path=tmp_path / "data" / "knowledge_base.csv",
        filter_date=processing_settings['filter_date'],
        sample_size=sample_settings['default_sample_size'],
        time_column=processing_settings['time_column'],
        strata_column=processing_settings['strata_column']
    )

@pytest.fixture
def setup_test_data(test_config):
    """Setup test data and directory structure."""
    # Create directories
    test_config.root_path.mkdir(parents=True, exist_ok=True)
    test_config.all_data_path.parent.mkdir(parents=True, exist_ok=True)
    test_config.stratified_data_path.mkdir(parents=True, exist_ok=True)
    test_config.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)

    # Create test DataFrame
    df = pd.DataFrame(TEST_DATA)
    
    # Save test data
    df.to_csv(test_config.all_data_path, index=False)
    
    yield test_config
    
    # Cleanup
    shutil.rmtree(test_config.root_path)

@pytest.mark.asyncio
async def test_initial_data_preparation(setup_test_data):
    """Test initial data preparation with force_refresh=True."""
    config = setup_test_data
    operations = DataOperations(config)
    
    # Test initial preparation
    result = await operations.prepare_data(force_refresh=True)
    assert "completed successfully" in result.lower()
    
    # Verify files were created
    assert config.knowledge_base_path.exists()
    assert (config.stratified_data_path / "stratified_sample.csv").exists()
    
    # Verify state was updated
    state_manager = DataStateManager(config)
    last_update = state_manager.get_last_update()
    assert last_update is not None

@pytest.mark.asyncio
async def test_incremental_update(setup_test_data):
    """Test incremental update with force_refresh=False."""
    config = setup_test_data
    operations = DataOperations(config)
    
    # Initial setup
    await operations.prepare_data(force_refresh=True)
    initial_state = DataStateManager(config).get_last_update()
    
    # Add new data
    new_data = pd.DataFrame({
        'thread_id': ['4', '5'],
        'posted_date_time': [
            (datetime.now(pytz.UTC) + timedelta(hours=1)).isoformat(),
            (datetime.now(pytz.UTC) + timedelta(hours=2)).isoformat()
        ],
        'text_clean': ['New content 1', 'New content 2'],
        'posted_comment': ['New comment 1', 'New comment 2']
    })
    
    # Append new data to existing file
    new_data.to_csv(config.all_data_path, mode='a', header=False, index=False)
    
    # Run incremental update
    result = await operations.prepare_data(force_refresh=False)
    assert "incremental update" in result.lower()
    
    # Verify state was updated
    new_state = DataStateManager(config).get_last_update()
    assert new_state > initial_state

@pytest.mark.asyncio
async def test_force_refresh_handling(setup_test_data):
    """Test force refresh handling in prepare_knowledge_base."""
    config = setup_test_data
    
    # Initial setup
    await prepare_knowledge_base(force_refresh=True)
    initial_modification_time = config.knowledge_base_path.stat().st_mtime
    
    # Try without force refresh
    await prepare_knowledge_base(force_refresh=False)
    assert config.knowledge_base_path.stat().st_mtime == initial_modification_time
    
    # Force refresh
    await prepare_knowledge_base(force_refresh=True)
    assert config.knowledge_base_path.stat().st_mtime > initial_modification_time

@pytest.mark.asyncio
async def test_embedding_generation(setup_test_data):
    """Test embedding generation and caching."""
    config = setup_test_data
    
    # Create test stratified data
    stratified_file = config.stratified_data_path / "stratified_sample.csv"
    pd.DataFrame(TEST_DATA).to_csv(stratified_file, index=False)
    
    # Initial embedding generation
    await get_relevant_content(
        library=str(config.stratified_data_path),
        knowledge_base=str(config.knowledge_base_path),
        batch_size=2,
        provider=ModelProvider.OPENAI,
        force_refresh=True
    )
    
    # Verify embeddings were generated
    kb_data = pd.read_csv(config.knowledge_base_path)
    assert 'embedding' in kb_data.columns
    assert len(kb_data) > 0
    
    # Test embedding caching
    initial_modification_time = config.knowledge_base_path.stat().st_mtime
    
    await get_relevant_content(
        library=str(config.stratified_data_path),
        knowledge_base=str(config.knowledge_base_path),
        batch_size=2,
        provider=ModelProvider.OPENAI,
        force_refresh=False
    )
    
    assert config.knowledge_base_path.stat().st_mtime == initial_modification_time

@pytest.mark.asyncio
async def test_data_state_management(setup_test_data):
    """Test data state management functionality."""
    config = setup_test_data
    state_manager = DataStateManager(config)
    
    # Test initial state
    assert state_manager.get_last_update() is None
    
    # Test state update
    state_manager.update_state(100)
    last_update = state_manager.get_last_update()
    assert last_update is not None
    
    # Test state persistence
    new_state_manager = DataStateManager(config)
    assert new_state_manager.get_last_update() == last_update

@pytest.mark.asyncio
async def test_error_handling(setup_test_data):
    """Test error handling in data operations."""
    config = setup_test_data
    operations = DataOperations(config)
    
    # Test invalid data handling
    invalid_data = pd.DataFrame({
        'wrong_column': ['test']
    })
    invalid_data.to_csv(config.all_data_path, index=False)
    
    with pytest.raises(Exception):
        await operations.prepare_data(force_refresh=True)

@pytest.mark.asyncio
async def test_scheduled_update(setup_test_data):
    """Test scheduled update functionality."""
    config = setup_test_data
    operations = DataOperations(config)
    
    # Initial setup
    await operations.prepare_data(force_refresh=True)
    initial_state = DataStateManager(config).get_last_update()
    
    # Simulate scheduled update with new data
    new_data = pd.DataFrame({
        'thread_id': ['6'],
        'posted_date_time': [(datetime.now(pytz.UTC) + timedelta(hours=3)).isoformat()],
        'text_clean': ['Scheduled update content'],
        'posted_comment': ['Scheduled comment']
    })
    new_data.to_csv(config.all_data_path, mode='a', header=False, index=False)
    
    # Run scheduled update
    result = await operations.prepare_data(force_refresh=False)
    assert "completed successfully" in result.lower()
    
    # Verify state was updated
    new_state = DataStateManager(config).get_last_update()
    assert new_state > initial_state

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 