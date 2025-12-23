"""Test suite for data ingestion functionality."""
import pytest
import pandas as pd
from pathlib import Path
import shutil
import logging
from datetime import datetime, timedelta
import pytz
import os

from knowledge_agents.data_ops import (
    DataConfig,
    DataOperations,
)
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
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create base paths
    root_path = tmp_path / "root"
    stratified_path = root_path / "stratified"
    temp_path = root_path / "temp"
    
    # Create directories
    root_path.mkdir(parents=True, exist_ok=True)
    stratified_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    return DataConfig(
        root_data_path=root_path,
        stratified_data_path=stratified_path,
        temp_path=temp_path,
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings.get('time_column', 'posted_date_time'),
        strata_column=column_settings.get('strata_column', 'thread_id')
    )

@pytest.fixture
def test_data_ops(test_config):
    """Setup DataOperations with test configuration."""
    # Remove async from the fixture definition
    data_ops = DataOperations(test_config)
    yield data_ops
    # Cleanup
    try:
        if test_config.root_data_path.exists():
            shutil.rmtree(test_config.root_data_path)
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

@pytest.fixture
def setup_test_data(test_config):
    """Setup test data and directory structure with proper directory creation and permissions."""
    # Ensure directories exist
    test_config.root_data_path.mkdir(parents=True, exist_ok=True)
    if test_config.stratified_data_path:
        test_config.stratified_data_path.mkdir(parents=True, exist_ok=True)
    
    # Try to set permissions
    try:
        os.chmod(str(test_config.root_data_path), 0o777)
        if test_config.stratified_data_path:
            os.chmod(str(test_config.stratified_data_path), 0o777)
    except Exception as e:
        logger.warning(f"Could not set permissions for directories: {e}")
    
    # Create test DataFrame
    df = pd.DataFrame(TEST_DATA)
    
    # Save test data to complete data file
    complete_data_path = test_config.root_data_path / "complete_data.csv"
    df.to_csv(complete_data_path, index=False)
    
    yield test_config
    
    # Cleanup handled by test_data_ops fixture

@pytest.mark.asyncio
async def test_initial_data_load(test_data_ops, setup_test_data):
    """Test initial data preparation with force_refresh=True."""
    # Arrange
    operations = test_data_ops
    
    # Act
    await operations.ensure_data_ready(force_refresh=True)
    
    # Assert
    complete_data_path = operations.config.root_data_path / "complete_data.csv"
    stratified_path = operations.config.stratified_data_path / "stratified_sample.csv"
    
    assert complete_data_path.exists(), "Complete data file should exist after test"
    assert stratified_path.exists(), "Stratified data file should exist after test"
    
    # Verify data content
    stratified_data = pd.read_csv(stratified_path)
    assert not stratified_data.empty, "Stratified data should not be empty"
    assert 'thread_id' in stratified_data.columns, "Stratified data should contain thread_id column"

@pytest.mark.asyncio
async def test_force_refresh_true(test_data_ops, setup_test_data):
    """Test data refresh with force_refresh=True."""
    # Arrange
    operations = test_data_ops
    
    # First load
    await operations.ensure_data_ready(force_refresh=False)
    initial_mtime = operations.config.stratified_data_path.stat().st_mtime
    
    # Act
    await operations.ensure_data_ready(force_refresh=True)
    
    # Assert
    new_mtime = operations.config.stratified_data_path.stat().st_mtime
    assert new_mtime >= initial_mtime, "Stratified data should be modified when force_refresh=true"

@pytest.mark.asyncio
async def test_force_refresh_false(test_data_ops, setup_test_data):
    """Test data handling with force_refresh=False."""
    # Arrange
    operations = test_data_ops
    
    # Act
    await operations.ensure_data_ready(force_refresh=False)
    
    # Assert
    complete_data_path = operations.config.root_data_path / "complete_data.csv"
    assert complete_data_path.exists(), "Complete data should be created if it didn't exist"

@pytest.mark.asyncio
async def test_embedding_generation(test_data_ops, setup_test_data):
    """Test embedding generation process."""
    # Arrange
    operations = test_data_ops
    
    # Act
    await operations.ensure_data_ready(force_refresh=True)
    await operations.generate_embeddings(force_refresh=True)
    
    # Assert
    embeddings_path = operations.config.stratified_data_path / "embeddings.npz"
    status_path = operations.config.stratified_data_path / "embedding_status.csv"
    
    assert embeddings_path.exists(), "Embeddings file should exist after test"
    assert status_path.exists(), "Embedding status file should exist after test"
    
    # Verify embedding status content
    status_df = pd.read_csv(status_path)
    assert not status_df.empty, "Embedding status should not be empty"
    assert 'thread_id' in status_df.columns, "Status should contain thread_id column"
    assert 'has_embedding' in status_df.columns, "Status should contain has_embedding column"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 
