import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import pytest
import logging
import sys
from datetime import datetime, timedelta
import pytz
import json
import os
from config.base_settings import load_env_vars

# Load base environment variables only
load_env_vars()

from knowledge_agents.data_ops import DataConfig, DataOperations
from knowledge_agents.model_ops import ModelConfig, ModelProvider, ModelOperation
from knowledge_agents.embedding_ops import KnowledgeDocument, get_relevant_content
from knowledge_agents.inference_ops import prepare_embeddings
from config.settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def test_data_config(tmp_path_factory):
    """Create a test configuration using pytest's tmp_path_factory."""
    # Create test directories using pytest's temporary directory
    test_root = tmp_path_factory.mktemp("test_data")
    test_stratified = test_root / "stratified"
    test_stratified.mkdir(exist_ok=True)
    test_temp = test_root / "temp"
    test_temp.mkdir(exist_ok=True)
    
    # Create test config
    config = DataConfig(
        root_data_path=test_root,
        stratified_data_path=test_stratified,
        temp_path=test_temp,
        sample_size=100
    )
    
    return config

def create_sample_data(size: int = 100) -> pd.DataFrame:
    """Create sample data for testing."""
    current_time = datetime.now(pytz.UTC)
    
    data = []
    for i in range(size):
        posted_time = current_time - timedelta(hours=i)
        # Use numeric thread IDs to match the format in the actual data
        data.append({
            'thread_id': str(10000000 + i),  # Numeric thread ID as string
            'posted_date_time': posted_time.isoformat(),
            'text_clean': f'This is test text for thread {i}',
            'posted_comment': f'Original comment for thread {i}'
        })
    
    return pd.DataFrame(data)

@pytest.mark.asyncio
async def test_embedding_pipeline(test_data_config):
    """Test the complete embedding pipeline."""
    try:
        logger.info("Starting embedding pipeline test")
        
        # Verify embedding provider availability (API key or mock embeddings)
        if not os.getenv('OPENAI_API_KEY') and not Config.use_mock_embeddings():
            pytest.skip("OPENAI_API_KEY is not set and USE_MOCK_EMBEDDINGS is disabled")
        
        # Step 1: Create and save sample data
        sample_df = create_sample_data()
        complete_data_path = test_data_config.root_data_path / 'complete_data.csv'
        complete_data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(complete_data_path, index=False)
        logger.info(f"Created sample data with {len(sample_df)} records")
        
        # Step 2: Initialize DataOperations
        data_ops = DataOperations(test_data_config)
        
        # Step 3: Test data preparation
        result = await data_ops.ensure_data_ready(force_refresh=True)
        assert result, "Data preparation failed"
        logger.info("Data preparation completed successfully")
        
        # Step 4: Verify stratified data
        stratified_file = test_data_config.stratified_data_path / 'stratified_sample.csv'
        assert stratified_file.exists(), "Stratified file not created"
        
        stratified_df = pd.read_csv(stratified_file)
        assert not stratified_df.empty, "Stratified data is empty"
        logger.info(f"Verified stratified data: {len(stratified_df)} records")
        
        # Step 5: Verify embeddings
        embeddings_path = test_data_config.stratified_data_path / 'embeddings.npz'
        thread_id_map_path = test_data_config.stratified_data_path / 'thread_id_map.json'
        
        assert embeddings_path.exists(), "Embeddings file not created"
        assert thread_id_map_path.exists(), "Thread ID map not created"
        
        # Load and verify embeddings
        with np.load(embeddings_path) as data:
            embeddings = data['embeddings']
        with open(thread_id_map_path, 'r') as f:
            thread_id_map = json.load(f)
            
        assert len(embeddings) == len(thread_id_map), "Mismatch between embeddings and thread IDs"
        logger.info(f"Verified embeddings: {len(embeddings)} embeddings created")
        
        # Step 6: Verify embedding status
        status_file = test_data_config.stratified_data_path / 'embedding_status.csv'
        assert status_file.exists(), "Embedding status file not created"
        
        status_df = pd.read_csv(status_file)
        assert not status_df.empty, "Embedding status is empty"
        logger.info(f"Verified embedding status: {len(status_df)} records")
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise 

def test_prepare_embeddings_various_formats():
    df = pd.DataFrame(
        {
            "embedding": [
                "[0.1, 0.0, 0.0]",  # JSON string
                [0.0, 0.2, 0.0],  # list
                np.array([0.0, 0.0, 0.3]),  # numpy array
                0.5,  # scalar
            ],
            "text_clean": ["a", "b", "c", "d"],
        }
    )
    query_emb = np.array([0.1, 0.2, 0.3])

    embeddings, indices, returned = prepare_embeddings(df, query_emb, adapt=False)
    assert embeddings.shape == (4, 3)
    assert indices == [0, 1, 2, 3]
    assert np.allclose(returned, query_emb)


def test_prepare_embeddings_dimension_adaptation():
    df = pd.DataFrame({"embedding": [[0.1], [0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]})
    query_emb = np.array([0.9, 0.8, 0.7])

    embeddings, indices, returned = prepare_embeddings(df, query_emb, adapt=True)
    assert embeddings.shape == (1, 3)
    assert indices == [2]
    assert np.allclose(returned, query_emb)
