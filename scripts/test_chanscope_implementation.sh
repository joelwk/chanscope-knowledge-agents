#!/bin/bash
set -e

# Enhanced test script to verify Chanscope implementation
# Designed to work in both Docker and Replit environments
echo "Testing Chanscope Implementation"
echo "==============================="

# Function to run a test
run_test() {
    local test_name=$1
    local command=$2
    echo
    echo "Running test: $test_name"
    echo "------------------------------"
    echo "Command: $command"
    echo
    eval "$command"
    local status=$?
    if [ $status -eq 0 ]; then
        echo "✅ Test passed: $test_name"
    else
        echo "❌ Test failed: $test_name (exit code: $status)"
    fi
    return $status
}

# Get the application root directory
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "App root: $APP_ROOT"

# Create test environment variables
export DATA_RETENTION_DAYS=7
export ENABLE_DATA_SCHEDULER=true
export DATA_UPDATE_INTERVAL=1800

# Detect environment
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment"
    ENV_TYPE="docker"
elif [ -n "$REPL_ID" ]; then
    echo "Running in Replit environment"
    ENV_TYPE="replit"
else
    echo "Running in local environment"
    ENV_TYPE="local"
fi

echo "Test configuration:"
echo "Environment: $ENV_TYPE"
echo "DATA_RETENTION_DAYS=$DATA_RETENTION_DAYS"
echo "ENABLE_DATA_SCHEDULER=$ENABLE_DATA_SCHEDULER"
echo "DATA_UPDATE_INTERVAL=$DATA_UPDATE_INTERVAL"

# Test 1: Test initial data load (force_refresh=true, skip_embeddings=true)
# This simulates the application startup behavior according to Chanscope approach
run_test "Initial Data Load" "cd $APP_ROOT && python -c \"
import asyncio
from knowledge_agents.data_ops import DataConfig, DataOperations
from config.settings import Config
from pathlib import Path

async def test_initial_load():
    # Get settings from Config
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create data config
    data_config = DataConfig(
        root_data_path=Path(paths['root_data_path']),
        stratified_data_path=Path(paths['stratified']),
        temp_path=Path(paths['temp']),
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings['time_column'],
        strata_column=column_settings['strata_column']
    )
    
    # Initialize data operations
    operations = DataOperations(data_config)
    
    # Test initial data load with force_refresh=True and skip_embeddings=True
    # This follows the Chanscope approach for application startup
    print('Testing initial data load (force_refresh=True, skip_embeddings=True)')
    result = await operations.ensure_data_ready(force_refresh=True, skip_embeddings=True)
    print(f'Result: {result}')
    
    # Verify that complete_data.csv exists
    complete_data_path = Path(paths['root_data_path']) / 'complete_data.csv'
    stratified_path = Path(paths['stratified']) / 'stratified_sample.csv'
    
    print(f'Verifying data files:')
    print(f'- Complete data exists: {complete_data_path.exists()}')
    print(f'- Stratified data exists: {stratified_path.exists()}')
    
    return result

asyncio.run(test_initial_load())
\""

# Test 2: Test embedding generation separately
# This validates the second phase of the Chanscope approach
run_test "Separate Embedding Generation" "cd $APP_ROOT && python -c \"
import asyncio
from knowledge_agents.data_ops import DataConfig, DataOperations
from config.settings import Config
from pathlib import Path

async def test_embedding_generation():
    # Get settings from Config
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create data config
    data_config = DataConfig(
        root_data_path=Path(paths['root_data_path']),
        stratified_data_path=Path(paths['stratified']),
        temp_path=Path(paths['temp']),
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings['time_column'],
        strata_column=column_settings['strata_column']
    )
    
    # Initialize data operations
    operations = DataOperations(data_config)
    
    # Test separate embedding generation
    print('Testing separate embedding generation')
    result = await operations.generate_embeddings(force_refresh=False)
    print(f'Result: {result}')
    
    # Verify that embeddings exist
    embeddings_path = Path(paths['stratified']) / 'embeddings.npz'
    thread_id_map_path = Path(paths['stratified']) / 'thread_id_map.json'
    
    print(f'Verifying embedding files:')
    print(f'- Embeddings exist: {embeddings_path.exists()}')
    print(f'- Thread ID map exists: {thread_id_map_path.exists()}')
    
    return result

asyncio.run(test_embedding_generation())
\""

# Test 3: Test force_refresh=false behavior
# This validates the incremental update behavior in Chanscope approach
run_test "Incremental Update (force_refresh=false)" "cd $APP_ROOT && python -c \"
import asyncio
import time
from knowledge_agents.data_ops import DataConfig, DataOperations
from config.settings import Config
from pathlib import Path

async def test_incremental_update():
    # Get settings from Config
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create data config
    data_config = DataConfig(
        root_data_path=Path(paths['root_data_path']),
        stratified_data_path=Path(paths['stratified']),
        temp_path=Path(paths['temp']),
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings['time_column'],
        strata_column=column_settings['strata_column']
    )
    
    # Initialize data operations
    operations = DataOperations(data_config)
    
    # Record file modification times before update
    complete_data_path = Path(paths['root_data_path']) / 'complete_data.csv'
    stratified_path = Path(paths['stratified']) / 'stratified_sample.csv'
    
    before_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    before_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    
    # Test incremental update with force_refresh=False
    print('Testing incremental update (force_refresh=False)')
    result = await operations.ensure_data_ready(force_refresh=False)
    print(f'Result: {result}')
    
    # Check if files were modified
    after_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    after_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    
    print(f'Verifying Chanscope force_refresh=false behavior:')
    print(f'- Complete data file modified: {before_complete_mtime != after_complete_mtime}')
    print(f'- Stratified data file modified: {before_stratified_mtime != after_stratified_mtime}')
    print(f'- According to Chanscope rules, files should NOT be modified unless missing')
    
    return result

asyncio.run(test_incremental_update())
\""

# Test 4: Test force_refresh=true behavior
# This validates the forced refresh behavior in Chanscope approach
run_test "Forced Refresh (force_refresh=true)" "cd $APP_ROOT && python -c \"
import asyncio
import time
from knowledge_agents.data_ops import DataConfig, DataOperations
from config.settings import Config
from pathlib import Path

async def test_forced_refresh():
    # Get settings from Config
    paths = Config.get_paths()
    processing_settings = Config.get_processing_settings()
    sample_settings = Config.get_sample_settings()
    column_settings = Config.get_column_settings()
    
    # Create data config
    data_config = DataConfig(
        root_data_path=Path(paths['root_data_path']),
        stratified_data_path=Path(paths['stratified']),
        temp_path=Path(paths['temp']),
        filter_date=processing_settings.get('filter_date'),
        sample_size=sample_settings['default_sample_size'],
        time_column=column_settings['time_column'],
        strata_column=column_settings['strata_column']
    )
    
    # Initialize data operations
    operations = DataOperations(data_config)
    
    # Record file modification times before update
    complete_data_path = Path(paths['root_data_path']) / 'complete_data.csv'
    stratified_path = Path(paths['stratified']) / 'stratified_sample.csv'
    embeddings_path = Path(paths['stratified']) / 'embeddings.npz'
    
    before_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    before_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    before_embeddings_mtime = embeddings_path.stat().st_mtime if embeddings_path.exists() else 0
    
    # Test forced refresh with force_refresh=True
    print('Testing forced refresh (force_refresh=True)')
    result = await operations.ensure_data_ready(force_refresh=True)
    print(f'Result: {result}')
    
    # Check if files were modified
    after_complete_mtime = complete_data_path.stat().st_mtime if complete_data_path.exists() else 0
    after_stratified_mtime = stratified_path.stat().st_mtime if stratified_path.exists() else 0
    after_embeddings_mtime = embeddings_path.stat().st_mtime if embeddings_path.exists() else 0
    
    print(f'Verifying Chanscope force_refresh=true behavior:')
    print(f'- Complete data file modified: {before_complete_mtime != after_complete_mtime}')
    print(f'- Stratified data file modified: {before_stratified_mtime != after_stratified_mtime}')
    print(f'- Embeddings file modified: {before_embeddings_mtime != after_embeddings_mtime}')
    print(f'- According to Chanscope rules, stratified data should ALWAYS be refreshed')
    
    return result

asyncio.run(test_forced_refresh())
\""

# Test 5: Test scheduled update in the current environment
# This validates the scheduled update functionality
run_test "Scheduled Update" "cd $APP_ROOT && python scripts/scheduled_update.py --run_once"

echo
echo "All tests completed"
echo "===============================" 

# Print summary of test results
echo "Test Summary:"
echo "1. Initial Data Load - Tests the application startup behavior"
echo "2. Separate Embedding Generation - Tests the second phase of startup"
echo "3. Incremental Update - Tests force_refresh=false behavior"
echo "4. Forced Refresh - Tests force_refresh=true behavior"
echo "5. Scheduled Update - Tests the scheduled update functionality"
echo
echo "Environment: $ENV_TYPE"
echo "These tests validate the Chanscope approach implementation" 