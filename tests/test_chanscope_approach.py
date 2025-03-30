#!/usr/bin/env python
"""
Chanscope Approach Validation Test

This pytest module validates that the implementation follows the Chanscope approach
as defined in approach-chanscope.mdc. It tests:

1. Initial data load behavior (force_refresh=true, skip_embeddings=true)
2. Separate embedding generation
3. force_refresh=false behavior
4. force_refresh=true behavior
"""

import os
import pytest
import logging
import sys
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Import centralized environment detection
from config.env_loader import detect_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load test environment variables
test_env_path = Path(__file__).parent / '.env.test'
if test_env_path.exists():
    load_dotenv(test_env_path, override=True)
    os.environ['USE_MOCK_EMBEDDINGS'] = 'true'
    os.environ['DEFAULT_EMBEDDING_PROVIDER'] = 'mock'
    logger.info(f"Loaded test environment from {test_env_path}")

# Import required modules
from knowledge_agents.data_ops import DataConfig, DataOperations
from config.settings import Config
from knowledge_agents.model_ops import ModelProvider

# Use centralized environment detection
ENV_TYPE = detect_environment()
logger.info(f"Detected environment: {ENV_TYPE}")

def create_mock_data(test_root: Path) -> None:
    """Create mock data files for testing."""
    # Create mock complete_data.csv
    complete_data_path = test_root / 'complete_data.csv'
    
    # Generate sample data
    sample_data = []
    for i in range(100):
        sample_data.append({
            'thread_id': str(10000000 + i),  # Use numeric thread IDs to match the format in the actual data
            'posted_date_time': datetime.now(pytz.UTC).isoformat(),
            'text_clean': f'This is sample text for article {i}. It contains information about a topic of interest.',
            'posted_comment': f'Comment for article {i}'
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(complete_data_path, index=False)
    logger.info(f"Created mock complete_data.csv with {len(df)} records at {complete_data_path}")

@pytest.fixture(scope="module")
def chanscope_test_config(tmp_path_factory):
    """Create a test configuration for Chanscope validation."""
    # Create test directories using pytest's temporary directory
    test_root = tmp_path_factory.mktemp("chanscope_test_data")
    test_stratified = test_root / "stratified"
    test_stratified.mkdir(exist_ok=True)
    try:
        os.chmod(str(test_stratified), 0o777)
    except Exception as e:
        logger.warning(f"Could not set permissions on {test_stratified}: {e}")
    test_temp = test_root / "temp"
    test_temp.mkdir(exist_ok=True)
    
    # Create mock data files
    create_mock_data(test_root)
    
    # Get configuration settings
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create test config
    config = DataConfig(
        root_data_path=test_root,
        stratified_data_path=test_stratified,
        temp_path=test_temp,
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings['time_column'],
        strata_column=column_settings['strata_column']
    )
    
    # Define file paths
    file_paths = {
        "complete_data_path": test_root / 'complete_data.csv',
        "stratified_path": test_stratified / 'stratified_sample.csv',
        "embeddings_path": test_stratified / 'embeddings.npz',
        "thread_id_map_path": test_stratified / 'thread_id_map.json'
    }
    
    # Create a DataOperations instance with the test config
    operations = DataOperations(config)
    
    # Override the paths in the operations object to ensure it uses the test paths
    operations.config.root_data_path = test_root
    operations.config.stratified_data_path = test_stratified
    operations.config.temp_path = test_temp
    
    return {
        "config": config,
        "paths": file_paths,
        "operations": operations
    }

@pytest.fixture(scope="module")
def results_collector():
    """Fixture to collect test results for reporting."""
    results = {
        "environment": ENV_TYPE,
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "tests": {}
    }
    
    yield results
    
    # After all tests, calculate overall compliance
    all_compliant = all(test.get("chanscope_compliant", False) for test in results["tests"].values())
    results["overall_chanscope_compliant"] = all_compliant
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"test_results/chanscope_validation_{ENV_TYPE}_{timestamp}.json"
    os.makedirs("test_results", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {output_path}")
    logger.info(f"Overall Chanscope compliance: {all_compliant}")

@pytest.mark.asyncio
async def test_initial_data_load(chanscope_test_config, results_collector):
    """
    Test 1: Initial data load (force_refresh=true, skip_embeddings=true)
    
    This test validates the first phase of the Chanscope approach:
    - Load data from S3 starting from DATA_RETENTION_DAYS ago
    - Save to complete_data.csv
    - Create stratified sample
    - Skip embedding generation
    """
    logger.info("Test 1: Initial data load (force_refresh=true, skip_embeddings=true)")
    
    # Get test configuration
    operations = chanscope_test_config["operations"]
    complete_data_path = chanscope_test_config["paths"]["complete_data_path"]
    stratified_path = chanscope_test_config["paths"]["stratified_path"]
    embeddings_path = chanscope_test_config["paths"]["embeddings_path"]
    
    # Record file existence before test
    before_complete_exists = complete_data_path.exists()
    before_stratified_exists = stratified_path.exists()
    before_embeddings_exist = embeddings_path.exists()
    
    # Record file modification times if they exist
    before_complete_mtime = complete_data_path.stat().st_mtime if before_complete_exists else 0
    before_stratified_mtime = stratified_path.stat().st_mtime if before_stratified_exists else 0
    
    start_time = time.time()
    
    # Run the test
    try:
        result = await operations.ensure_data_ready(force_refresh=True, skip_embeddings=True)
        success = True
    except Exception as e:
        logger.error(f"Error in initial data load test: {e}", exc_info=True)
        result = str(e)
        success = False
    
    duration = time.time() - start_time
    
    # Check file existence after test
    after_complete_exists = complete_data_path.exists()
    after_stratified_exists = stratified_path.exists()
    after_embeddings_exist = embeddings_path.exists()
    
    # Record file modification times if they exist
    after_complete_mtime = complete_data_path.stat().st_mtime if after_complete_exists else 0
    after_stratified_mtime = stratified_path.stat().st_mtime if after_stratified_exists else 0
    
    # Determine if files were modified
    complete_modified = before_complete_mtime != after_complete_mtime
    stratified_modified = before_stratified_mtime != after_stratified_mtime
    
    # Prepare test results
    test_results = {
        "success": success,
        "duration_seconds": duration,
        "result": result,
        "before": {
            "complete_data_exists": before_complete_exists,
            "stratified_data_exists": before_stratified_exists,
            "embeddings_exist": before_embeddings_exist
        },
        "after": {
            "complete_data_exists": after_complete_exists,
            "stratified_data_exists": after_stratified_exists,
            "embeddings_exist": after_embeddings_exist,
            "complete_data_modified": complete_modified,
            "stratified_data_modified": stratified_modified
        },
        "chanscope_compliant": after_complete_exists and after_stratified_exists and not after_embeddings_exist
    }
    
    # Log results
    logger.info(f"Initial data load test completed in {duration:.2f} seconds")
    logger.info(f"Complete data exists: {after_complete_exists}")
    logger.info(f"Stratified data exists: {after_stratified_exists}")
    logger.info(f"Embeddings exist: {after_embeddings_exist}")
    logger.info(f"Chanscope compliant: {test_results['chanscope_compliant']}")
    
    # Store results
    results_collector["tests"]["initial_data_load"] = test_results
    
    # Assert test conditions
    assert after_complete_exists, "Complete data file should exist after test"
    assert after_stratified_exists, "Stratified data file should exist after test"
    assert not after_embeddings_exist, "Embeddings should not exist after test (skip_embeddings=True)"
    assert test_results["chanscope_compliant"], "Test should be Chanscope compliant"

@pytest.mark.asyncio
async def test_embedding_generation(chanscope_test_config, results_collector):
    """
    Test 2: Separate embedding generation
    
    This test validates the second phase of the Chanscope approach:
    - Generate embeddings from the stratified data
    """
    logger.info("Test 2: Separate embedding generation")
    
    # Get test configuration
    operations = chanscope_test_config["operations"]
    embeddings_path = chanscope_test_config["paths"]["embeddings_path"]
    thread_id_map_path = chanscope_test_config["paths"]["thread_id_map_path"]
    
    # Record file existence before test
    before_embeddings_exist = embeddings_path.exists()
    before_thread_id_map_exists = thread_id_map_path.exists()
    
    # Record file modification times if they exist
    before_embeddings_mtime = embeddings_path.stat().st_mtime if before_embeddings_exist else 0
    before_thread_id_map_mtime = thread_id_map_path.stat().st_mtime if before_thread_id_map_exists else 0
    
    start_time = time.time()
    
    # Run the test
    try:
        result = await operations.generate_embeddings(force_refresh=False)
        success = result.get("success", False)
    except Exception as e:
        logger.error(f"Error in embedding generation test: {e}", exc_info=True)
        result = str(e)
        success = False
    
    duration = time.time() - start_time
    
    # Check file existence after test
    after_embeddings_exist = embeddings_path.exists()
    after_thread_id_map_exists = thread_id_map_path.exists()
    
    # Record file modification times if they exist
    after_embeddings_mtime = embeddings_path.stat().st_mtime if after_embeddings_exist else 0
    after_thread_id_map_mtime = thread_id_map_path.stat().st_mtime if after_thread_id_map_exists else 0
    
    # Determine if files were modified
    embeddings_modified = before_embeddings_mtime != after_embeddings_mtime
    thread_id_map_modified = before_thread_id_map_mtime != after_thread_id_map_mtime
    
    # Prepare test results
    test_results = {
        "success": success,
        "duration_seconds": duration,
        "result": result,
        "before": {
            "embeddings_exist": before_embeddings_exist,
            "thread_id_map_exists": before_thread_id_map_exists
        },
        "after": {
            "embeddings_exist": after_embeddings_exist,
            "thread_id_map_exists": after_thread_id_map_exists,
            "embeddings_modified": embeddings_modified,
            "thread_id_map_modified": thread_id_map_modified
        },
        "chanscope_compliant": after_embeddings_exist and after_thread_id_map_exists
    }
    
    # Log results
    logger.info(f"Embedding generation test completed in {duration:.2f} seconds")
    logger.info(f"Embeddings exist: {after_embeddings_exist}")
    logger.info(f"Thread ID map exists: {after_thread_id_map_exists}")
    logger.info(f"Chanscope compliant: {test_results['chanscope_compliant']}")
    
    # Store results
    results_collector["tests"]["embedding_generation"] = test_results
    
    # Assert test conditions
    assert after_embeddings_exist, "Embeddings file should exist after test"
    assert after_thread_id_map_exists, "Thread ID map should exist after test"
    assert test_results["chanscope_compliant"], "Test should be Chanscope compliant"

@pytest.mark.asyncio
async def test_force_refresh_false(chanscope_test_config, results_collector):
    """
    Test 3: force_refresh=false behavior
    
    This test validates the behavior when force_refresh=false:
    - If files exist, they should NOT be modified
    - If files don't exist, they should be created (behave like force_refresh=true)
    """
    logger.info("Test 3: force_refresh=false behavior")
    
    # Get test configuration
    operations = chanscope_test_config["operations"]
    complete_data_path = chanscope_test_config["paths"]["complete_data_path"]
    stratified_path = chanscope_test_config["paths"]["stratified_path"]
    embeddings_path = chanscope_test_config["paths"]["embeddings_path"]
    
    # Record file existence and modification times before test
    before_complete_exists = complete_data_path.exists()
    before_stratified_exists = stratified_path.exists()
    before_embeddings_exist = embeddings_path.exists()
    
    before_complete_mtime = complete_data_path.stat().st_mtime if before_complete_exists else 0
    before_stratified_mtime = stratified_path.stat().st_mtime if before_stratified_exists else 0
    before_embeddings_mtime = embeddings_path.stat().st_mtime if before_embeddings_exist else 0
    
    start_time = time.time()
    
    # Run the test
    try:
        result = await operations.ensure_data_ready(force_refresh=False)
        success = True
    except Exception as e:
        logger.error(f"Error in force_refresh=false test: {e}", exc_info=True)
        result = str(e)
        success = False
    
    duration = time.time() - start_time
    
    # Check file modification times after test
    after_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    after_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    after_embeddings_mtime = embeddings_path.stat().st_mtime if embeddings_path.exists() else 0
    
    # Determine if files were modified
    complete_modified = before_complete_mtime != after_complete_mtime
    stratified_modified = before_stratified_mtime != after_stratified_mtime
    embeddings_modified = before_embeddings_mtime != after_embeddings_mtime
    
    # According to Chanscope, with force_refresh=false:
    # - If files exist, they should NOT be modified
    # - If files don't exist, they should be created (behave like force_refresh=true)
    files_existed_before = before_complete_exists and before_stratified_exists and before_embeddings_exist
    
    if files_existed_before:
        # If files existed, they should not be modified
        chanscope_compliant = not complete_modified and not stratified_modified and not embeddings_modified
    else:
        # If files didn't exist, they should be created
        chanscope_compliant = complete_data_path.exists() and stratified_path.exists() and embeddings_path.exists()
    
    # Prepare test results
    test_results = {
        "success": success,
        "duration_seconds": duration,
        "result": result,
        "files_existed_before": files_existed_before,
        "modifications": {
            "complete_data_modified": complete_modified,
            "stratified_data_modified": stratified_modified,
            "embeddings_modified": embeddings_modified
        },
        "chanscope_compliant": chanscope_compliant
    }
    
    # Log results
    logger.info(f"force_refresh=false test completed in {duration:.2f} seconds")
    logger.info(f"Files existed before: {files_existed_before}")
    logger.info(f"Complete data modified: {complete_modified}")
    logger.info(f"Stratified data modified: {stratified_modified}")
    logger.info(f"Embeddings modified: {embeddings_modified}")
    logger.info(f"Chanscope compliant: {chanscope_compliant}")
    
    # Store results
    results_collector["tests"]["force_refresh_false"] = test_results
    
    # Assert test conditions
    if files_existed_before:
        assert not complete_modified, "Complete data should not be modified when force_refresh=false"
        assert not stratified_modified, "Stratified data should not be modified when force_refresh=false"
        assert not embeddings_modified, "Embeddings should not be modified when force_refresh=false"
    else:
        assert complete_data_path.exists(), "Complete data should be created if it didn't exist"
        assert stratified_path.exists(), "Stratified data should be created if it didn't exist"
        assert embeddings_path.exists(), "Embeddings should be created if they didn't exist"
    
    assert test_results["chanscope_compliant"], "Test should be Chanscope compliant"

@pytest.mark.asyncio
async def test_force_refresh_true(chanscope_test_config, results_collector):
    """
    Test 4: force_refresh=true behavior
    
    This test validates the behavior when force_refresh=true:
    - Complete data should only be refreshed if not up-to-date
    - Stratified data should ALWAYS be refreshed
    - Embeddings should ALWAYS be refreshed
    """
    logger.info("Test 4: force_refresh=true behavior")
    
    # Get test configuration
    operations = chanscope_test_config["operations"]
    complete_data_path = chanscope_test_config["paths"]["complete_data_path"]
    stratified_path = chanscope_test_config["paths"]["stratified_path"]
    embeddings_path = chanscope_test_config["paths"]["embeddings_path"]
    
    # Record file modification times before test
    before_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    before_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    before_embeddings_mtime = embeddings_path.stat().st_mtime if embeddings_path.exists() else 0
    
    start_time = time.time()
    
    # Run the test
    try:
        result = await operations.ensure_data_ready(force_refresh=True)
        success = True
    except Exception as e:
        logger.error(f"Error in force_refresh=true test: {e}", exc_info=True)
        result = str(e)
        success = False
    
    duration = time.time() - start_time
    
    # Check file modification times after test
    after_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    after_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    after_embeddings_mtime = embeddings_path.stat().st_mtime if embeddings_path.exists() else 0
    
    # Determine if files were modified
    complete_modified = before_complete_mtime != after_complete_mtime
    stratified_modified = before_stratified_mtime != after_stratified_mtime
    embeddings_modified = before_embeddings_mtime != after_embeddings_mtime
    
    # According to Chanscope, with force_refresh=true:
    # - Complete data should only be refreshed if not up-to-date
    # - Stratified data should ALWAYS be refreshed
    # - Embeddings should ALWAYS be refreshed
    chanscope_compliant = stratified_modified and embeddings_modified
    
    # Prepare test results
    test_results = {
        "success": success,
        "duration_seconds": duration,
        "result": result,
        "modifications": {
            "complete_data_modified": complete_modified,
            "stratified_data_modified": stratified_modified,
            "embeddings_modified": embeddings_modified
        },
        "chanscope_compliant": chanscope_compliant
    }
    
    # Log results
    logger.info(f"force_refresh=true test completed in {duration:.2f} seconds")
    logger.info(f"Complete data modified: {complete_modified}")
    logger.info(f"Stratified data modified: {stratified_modified}")
    logger.info(f"Embeddings modified: {embeddings_modified}")
    logger.info(f"Chanscope compliant: {chanscope_compliant}")
    
    # Store results
    results_collector["tests"]["force_refresh_true"] = test_results
    
    # Assert test conditions
    assert stratified_modified, "Stratified data should be modified when force_refresh=true"
    assert embeddings_modified, "Embeddings should be modified when force_refresh=true"
    assert test_results["chanscope_compliant"], "Test should be Chanscope compliant" 