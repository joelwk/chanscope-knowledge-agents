#!/usr/bin/env python3
"""
Data Processing Utility for Chanscope

This script provides direct access to the Chanscope data processing pipeline,
allowing manual triggering of data loading, stratification, and embedding generation.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Ensure python path includes the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from knowledge_agents.data_processing.chanscope_manager import ChanScopeDataManager


async def process_data(force_refresh: bool = False, skip_embeddings: bool = False):
    """
    Process the data according to the Chanscope approach.
    
    Args:
        force_refresh: Whether to force refresh all data
        skip_embeddings: Whether to skip embedding generation
    """
    print(f"Starting data processing (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})...")
    
    try:
        # Create data manager
        data_manager = ChanScopeDataManager()
        
        # Process data
        data_ready = await data_manager.ensure_data_ready(
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
        
        if data_ready:
            print("Data processing completed successfully!")
            print("Data is ready for use (includes stratification and embeddings)")
            return True
        else:
            print("Warning: Data could not be fully prepared")
            return False
            
    except Exception as e:
        print(f"Error during data processing: {e}")
        return False


async def check_data_status():
    """Check the current status of the data processing pipeline."""
    try:
        # Create data manager
        data_manager = ChanScopeDataManager()
        
        # Check if complete data exists
        if hasattr(data_manager, 'complete_data_storage'):
            row_count = await data_manager.complete_data_storage.get_row_count()
            print(f"Complete data rows: {row_count}")
        else:
            print("Complete data storage not available")
        
        # Check if stratified data exists
        if hasattr(data_manager, 'stratified_data_storage'):
            strat_row_count = await data_manager.stratified_data_storage.get_row_count()
            print(f"Stratified data rows: {strat_row_count}")
        else:
            print("Stratified data storage not available")
        
        # Check if embeddings exist
        embeddings_available = await data_manager.check_embeddings_available()
        print(f"Embeddings available: {embeddings_available}")
        
        # Return overall status
        return row_count > 0 and strat_row_count > 0 and embeddings_available
        
    except Exception as e:
        print(f"Error checking data status: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chanscope Data Processing Utility")
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check data status without processing"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh all data"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # For Replit, ensure environment is properly set
    if os.environ.get('REPLIT_ENV') or os.environ.get('REPL_ID'):
        print("Running in Replit environment")
        if not os.environ.get('REPLIT_ENV'):
            os.environ['REPLIT_ENV'] = 'replit'
    
    # Either check status or process data
    if args.check:
        print("Checking data status...")
        asyncio.run(check_data_status())
    else:
        print("Processing data...")
        asyncio.run(process_data(
            force_refresh=args.force_refresh,
            skip_embeddings=args.skip_embeddings
        )) 