#!/usr/bin/env python
"""
Chanscope Approach Validation Script

This script validates that the implementation follows the Chanscope approach
as defined in approach-chanscope.mdc. It tests:

1. Initial data load behavior (force_refresh=true, skip_embeddings=true)
2. Separate embedding generation
3. force_refresh=false behavior
4. force_refresh=true behavior
5. Query processing pipeline

The script is designed to work in Docker, Replit, and local environments.
"""

import os
import sys
import asyncio
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List, Tuple

# Import environment detection from the centralized location
from config.env_loader import detect_environment

# Set environment variables for testing
os.environ['USE_MOCK_EMBEDDINGS'] = 'true'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use the centralized environment detection
ENV_TYPE = detect_environment()
logger.info(f"Detected environment: {ENV_TYPE}")

class ChanScopeValidator:
    """Validates the Chanscope approach implementation."""
    
    def __init__(self):
        """Initialize the validator."""
        # Import dependencies here to avoid import errors
        from config.settings import Config
        from knowledge_agents.data_ops import DataConfig, DataOperations
        
        self.Config = Config
        self.DataConfig = DataConfig
        self.DataOperations = DataOperations
        
        # Get configuration
        self.paths = Config.get_paths()
        self.processing_settings = Config.get_processing_settings()
        self.sample_settings = Config.get_sample_settings()
        self.column_settings = Config.get_column_settings()
        
        # Initialize data config
        self.data_config = DataConfig(
            root_data_path=Path(self.paths['root_data_path']),
            stratified_data_path=Path(self.paths['stratified']),
            temp_path=Path(self.paths['temp']),
            filter_date=self.processing_settings.get('filter_date'),
            sample_size=self.sample_settings['default_sample_size'],
            time_column=self.column_settings['time_column'],
            strata_column=self.column_settings['strata_column']
        )
        
        # Initialize data operations
        self.operations = DataOperations(self.data_config)
        
        # Define file paths
        self.complete_data_path = Path(self.paths['root_data_path']) / 'complete_data.csv'
        self.stratified_path = Path(self.paths['stratified']) / 'stratified_sample.csv'
        self.embeddings_path = Path(self.paths['stratified']) / 'embeddings.npz'
        self.thread_id_map_path = Path(self.paths['stratified']) / 'thread_id_map.json'
        
        # Create results dictionary
        self.results = {
            "environment": ENV_TYPE,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "tests": {}
        }
    
    async def test_initial_data_load(self) -> Dict[str, Any]:
        """
        Test 1: Initial data load (force_refresh=true, skip_embeddings=true)
        
        This test validates the first phase of the Chanscope approach:
        - Load data from S3 starting from DATA_RETENTION_DAYS ago
        - Save to complete_data.csv
        - Create stratified sample
        - Skip embedding generation
        """
        logger.info("Test 1: Initial data load (force_refresh=true, skip_embeddings=true)")
        
        # Record file existence before test
        before_complete_exists = self.complete_data_path.exists()
        before_stratified_exists = self.stratified_path.exists()
        before_embeddings_exist = self.embeddings_path.exists()
        
        # Record file modification times if they exist
        before_complete_mtime = self.complete_data_path.stat().st_mtime if before_complete_exists else 0
        before_stratified_mtime = self.stratified_path.stat().st_mtime if before_stratified_exists else 0
        
        start_time = time.time()
        
        # Run the test
        try:
            result = await self.operations.ensure_data_ready(force_refresh=True, skip_embeddings=True)
            success = True
        except Exception as e:
            logger.error(f"Error in initial data load test: {e}", exc_info=True)
            result = str(e)
            success = False
        
        duration = time.time() - start_time
        
        # Check file existence after test
        after_complete_exists = self.complete_data_path.exists()
        after_stratified_exists = self.stratified_path.exists()
        after_embeddings_exist = self.embeddings_path.exists()
        
        # Record file modification times if they exist
        after_complete_mtime = self.complete_data_path.stat().st_mtime if after_complete_exists else 0
        after_stratified_mtime = self.stratified_path.stat().st_mtime if after_stratified_exists else 0
        
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
        self.results["tests"]["initial_data_load"] = test_results
        
        return test_results
    
    async def test_embedding_generation(self) -> Dict[str, Any]:
        """
        Test 2: Separate embedding generation
        
        This test validates the second phase of the Chanscope approach:
        - Generate embeddings from the stratified data
        """
        logger.info("Test 2: Separate embedding generation")
        
        # Record file existence before test
        before_embeddings_exist = self.embeddings_path.exists()
        before_thread_id_map_exists = self.thread_id_map_path.exists()
        
        # Record file modification times if they exist
        before_embeddings_mtime = self.embeddings_path.stat().st_mtime if before_embeddings_exist else 0
        before_thread_id_map_mtime = self.thread_id_map_path.stat().st_mtime if before_thread_id_map_exists else 0
        
        start_time = time.time()
        
        # Run the test
        try:
            result = await self.operations.generate_embeddings(force_refresh=False)
            success = result.get("success", False)
        except Exception as e:
            logger.error(f"Error in embedding generation test: {e}", exc_info=True)
            result = str(e)
            success = False
        
        duration = time.time() - start_time
        
        # Check file existence after test
        after_embeddings_exist = self.embeddings_path.exists()
        after_thread_id_map_exists = self.thread_id_map_path.exists()
        
        # Record file modification times if they exist
        after_embeddings_mtime = self.embeddings_path.stat().st_mtime if after_embeddings_exist else 0
        after_thread_id_map_mtime = self.thread_id_map_path.stat().st_mtime if after_thread_id_map_exists else 0
        
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
        self.results["tests"]["embedding_generation"] = test_results
        
        return test_results
    
    async def test_force_refresh_false(self) -> Dict[str, Any]:
        """
        Test 3: force_refresh=false behavior
        
        This test validates the Chanscope approach for force_refresh=false:
        - Check if complete_data.csv exists, not whether it's fresh
        - Skip creating new stratified data and embeddings unless completely missing
        """
        logger.info("Test 3: force_refresh=false behavior")
        
        # Record file modification times before test
        before_complete_mtime = self.complete_data_path.stat().st_mtime if self.complete_data_path.exists() else 0
        before_stratified_mtime = self.stratified_path.stat().st_mtime if self.stratified_path.exists() else 0
        before_embeddings_mtime = self.embeddings_path.stat().st_mtime if self.embeddings_path.exists() else 0
        
        start_time = time.time()
        
        # Run the test
        try:
            result = await self.operations.ensure_data_ready(force_refresh=False)
            success = True
        except Exception as e:
            logger.error(f"Error in force_refresh=false test: {e}", exc_info=True)
            result = str(e)
            success = False
        
        duration = time.time() - start_time
        
        # Record file modification times after test
        after_complete_mtime = self.complete_data_path.stat().st_mtime if self.complete_data_path.exists() else 0
        after_stratified_mtime = self.stratified_path.stat().st_mtime if self.stratified_path.exists() else 0
        after_embeddings_mtime = self.embeddings_path.stat().st_mtime if self.embeddings_path.exists() else 0
        
        # Determine if files were modified
        complete_modified = before_complete_mtime != after_complete_mtime
        stratified_modified = before_stratified_mtime != after_stratified_mtime
        embeddings_modified = before_embeddings_mtime != after_embeddings_mtime
        
        # According to Chanscope, with force_refresh=false:
        # - If files exist, they should NOT be modified
        # - If files don't exist, they should be created (behave like force_refresh=true)
        files_existed_before = self.complete_data_path.exists() and self.stratified_path.exists() and self.embeddings_path.exists()
        if files_existed_before:
            # If files existed, they should not be modified
            chanscope_compliant = not complete_modified and not stratified_modified and not embeddings_modified
        else:
            # If files didn't exist, they should be created
            chanscope_compliant = self.complete_data_path.exists() and self.stratified_path.exists() and self.embeddings_path.exists()
        
        # Prepare test results
        test_results = {
            "success": success,
            "duration_seconds": duration,
            "result": result,
            "files_existed_before": files_existed_before,
            "modifications": {
                "complete_data_modified": complete_modified,
                "stratified_data_modified": stratified_modified,
                "embeddings_modified": embeddings_modified},
            "chanscope_compliant": chanscope_compliant}
        
        # Log results
        logger.info(f"force_refresh=false test completed in {duration:.2f} seconds")
        logger.info(f"Files existed before: {files_existed_before}")
        logger.info(f"Complete data modified: {complete_modified}")
        logger.info(f"Stratified data modified: {stratified_modified}")
        logger.info(f"Embeddings modified: {embeddings_modified}")
        logger.info(f"Chanscope compliant: {chanscope_compliant}")
        
        # Store results
        self.results["tests"]["force_refresh_false"] = test_results
        
        return test_results
    
    async def test_force_refresh_true(self) -> Dict[str, Any]:
        """
        Test 4: force_refresh=true behavior
        
        This test validates the Chanscope approach for force_refresh=true:
        - Verify that complete_data.csv exists and is up-to-date
        - Always create new stratified sample
        - Always generate new embeddings
        """
        logger.info("Test 4: force_refresh=true behavior")
        
        # Record file modification times before test
        before_complete_mtime = self.complete_data_path.stat().st_mtime if self.complete_data_path.exists() else 0
        before_stratified_mtime = self.stratified_path.stat().st_mtime if self.stratified_path.exists() else 0
        before_embeddings_mtime = self.embeddings_path.stat().st_mtime if self.embeddings_path.exists() else 0
        start_time = time.time()
        
        # Run the test
        try:
            result = await self.operations.ensure_data_ready(force_refresh=True)
            success = True
        except Exception as e:
            logger.error(f"Error in force_refresh=true test: {e}", exc_info=True)
            result = str(e)
            success = False
        duration = time.time() - start_time
        
        # Record file modification times after test
        after_complete_mtime = self.complete_data_path.stat().st_mtime if self.complete_data_path.exists() else 0
        after_stratified_mtime = self.stratified_path.stat().st_mtime if self.stratified_path.exists() else 0
        after_embeddings_mtime = self.embeddings_path.stat().st_mtime if self.embeddings_path.exists() else 0
        
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
                "embeddings_modified": embeddings_modified},
            "chanscope_compliant": chanscope_compliant}
        
        # Log results
        logger.info(f"force_refresh=true test completed in {duration:.2f} seconds")
        logger.info(f"Complete data modified: {complete_modified}")
        logger.info(f"Stratified data modified: {stratified_modified}")
        logger.info(f"Embeddings modified: {embeddings_modified}")
        logger.info(f"Chanscope compliant: {chanscope_compliant}")
        
        # Store results
        self.results["tests"]["force_refresh_true"] = test_results
        
        return test_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        logger.info(f"Running all Chanscope validation tests in {ENV_TYPE} environment")
        
        # Run tests in sequence
        await self.test_initial_data_load()
        await self.test_embedding_generation()
        await self.test_force_refresh_false()
        await self.test_force_refresh_true()
        
        # Calculate overall compliance
        all_compliant = all(test.get("chanscope_compliant", False) for test in self.results["tests"].values())
        self.results["overall_chanscope_compliant"] = all_compliant
        
        logger.info(f"All tests completed. Overall Chanscope compliance: {all_compliant}")
        
        return self.results
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save test results to a JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"chanscope_validation_{ENV_TYPE}_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to {output_path}")
        return output_path

async def main():
    """Main function to run the validation script."""
    parser = argparse.ArgumentParser(description='Validate Chanscope approach implementation')
    parser.add_argument('--output', help='Output file path for test results')
    args = parser.parse_args()
    
    try:
        validator = ChanScopeValidator()
        results = await validator.run_all_tests()
        output_path = validator.save_results(args.output)
        
        # Print summary
        print("\nChanscope Validation Summary:")
        print("============================")
        print(f"Environment: {ENV_TYPE}")
        print(f"Overall compliance: {'✅ PASS' if results['overall_chanscope_compliant'] else '❌ FAIL'}")
        print("\nTest Results:")
        for test_name, test_result in results["tests"].items():
            status = "✅ PASS" if test_result.get("chanscope_compliant", False) else "❌ FAIL"
            print(f"- {test_name}: {status}")
        print(f"\nDetailed results saved to: {output_path}")
        
        # Exit with appropriate status code
        sys.exit(0 if results["overall_chanscope_compliant"] else 1)
        
    except Exception as e:
        logger.error(f"Error running validation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 