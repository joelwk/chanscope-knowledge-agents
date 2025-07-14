#!/usr/bin/env python3
"""
Data Processing Utility for Chanscope

This script provides direct access to the Chanscope data processing pipeline,
allowing manual triggering of data loading, stratification, and embedding generation.
Includes text validation to ensure high-quality data throughout the pipeline.
"""

import os
import sys
import asyncio
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure python path includes the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from knowledge_agents.data_processing.chanscope_manager import ChanScopeDataManager
from config.chanscope_config import ChanScopeConfig
from scripts.utils.processing_lock import ProcessLockManager


async def process_data(force_refresh: bool = False, skip_embeddings: bool = False):
    """
    Process the data according to the Chanscope approach.
    
    Args:
        force_refresh: Whether to force refresh all data
        skip_embeddings: Whether to skip embedding generation
    """
    print(f"Starting data processing (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})...")
    print("Text validation will be applied at each stage to ensure high-quality data")
    
    # Create the process lock manager
    lock_manager = ProcessLockManager()
    
    # Try to acquire the processing lock
    if not lock_manager.acquire_lock():
        print("Another data processing instance is already running. Exiting.")
        return False
    
    try:
        # Create configuration
        config = ChanScopeConfig.from_env()
        
        # Override with function arguments
        if force_refresh:
            config.force_refresh = force_refresh
            
        # Create data manager with the config
        data_manager = ChanScopeDataManager(config)
        
        # Process data
        data_ready = await data_manager.ensure_data_ready(
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
        
        if data_ready:
            print("Data processing completed successfully!")
            
            # In Replit environment, verify the stratified sample is properly stored
            if os.environ.get('REPLIT_ENV') or os.environ.get('REPL_ID'):
                print("Verifying stratified sample in Replit key-value store...")
                # Check if stratified sample exists
                strat_exists = await data_manager.stratified_storage.sample_exists()
                if not strat_exists:
                    print("Stratified sample not found in key-value store, forcing creation...")
                    # Force stratified sample creation
                    await data_manager.create_stratified_sample(force_refresh=True)
                    # Verify again
                    strat_exists = await data_manager.stratified_storage.sample_exists()
                    
                if strat_exists:
                    # Try to load sample
                    sample = await data_manager.stratified_storage.get_sample()
                    if sample is not None and len(sample) > 0:
                        print(f"✅ Verified stratified sample in key-value store: {len(sample)} rows")
                    else:
                        print("⚠️ Warning: Stratified sample exists but couldn't be loaded")
                else:
                    print("❌ Error: Failed to create stratified sample in key-value store")
            
            # Mark initialization as complete
            lock_manager.mark_initialization_complete(True, {
                "force_refresh": force_refresh,
                "skip_embeddings": skip_embeddings
            })
            
            print("Data is ready for use (includes stratification and embeddings)")
            return True
        else:
            print("Warning: Data could not be fully prepared")
            lock_manager.mark_initialization_complete(False, {
                "error": "Data not fully prepared",
                "force_refresh": force_refresh,
                "skip_embeddings": skip_embeddings
            })
            return False
            
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Mark initialization as failed
        lock_manager.mark_initialization_complete(False, {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "force_refresh": force_refresh,
            "skip_embeddings": skip_embeddings
        })
        
        return False
    
    finally:
        # Always release the lock when done
        lock_manager.release_lock()


async def check_data_status():
    """Check the current status of the data processing pipeline."""
    try:
        # Create configuration
        config = ChanScopeConfig.from_env()
        
        # Create data manager with the config
        data_manager = ChanScopeDataManager(config)
        
        # Explicitly initialize PostgreSQL schema if in Replit environment
        if os.environ.get('REPLIT_ENV') or os.environ.get('REPL_ID'):
            print("Initializing PostgreSQL schema...")
            from config.replit import PostgresDB
            try:
                db = PostgresDB()
                db.initialize_schema()
                print("PostgreSQL schema initialized successfully")
            except Exception as e:
                print(f"Error initializing PostgreSQL schema: {e}")
        
        # Check if complete data exists
        if hasattr(data_manager, 'complete_data_storage'):
            row_count = await data_manager.complete_data_storage.get_row_count()
            print(f"Complete data rows: {row_count}")
        else:
            print("Complete data storage not available")
        
        # Check if stratified data exists
        if hasattr(data_manager, 'stratified_storage'):
            strat_exists = await data_manager.stratified_storage.sample_exists()
            print(f"Stratified data exists: {strat_exists}")
            if strat_exists:
                strat_sample = await data_manager.stratified_storage.get_sample()
                if strat_sample is not None:
                    print(f"Stratified data rows: {len(strat_sample)}")
                else:
                    print("Stratified data exists but couldn't be loaded")
            else:
                print("Stratified data doesn't exist")
        else:
            print("Stratified data storage not available")
        
        # Check if embeddings exist
        embeddings_available = await data_manager.check_embeddings_available() if hasattr(data_manager, 'check_embeddings_available') else False
        print(f"Embeddings available: {embeddings_available}")
        
        # Check storage backend type
        if hasattr(data_manager, 'embedding_storage'):
            storage_type = data_manager.embedding_storage.__class__.__name__
            print(f"Embedding storage backend: {storage_type}")
            
            # Verify Object Storage is being used in Replit environment
            if os.environ.get('REPLIT_ENV') or os.environ.get('REPL_ID'):
                if "Object" in storage_type:
                    print("✅ Using Object Storage for embeddings (recommended for large embeddings)")
                else:
                    print("⚠️ WARNING: Not using Object Storage for embeddings. Large embeddings may fail to store.")
        
        # Check initialization status
        lock_manager = ProcessLockManager()
        needs_init, marker_data = lock_manager.check_initialization_status()
        if marker_data:
            print(f"Initialization status: {marker_data.get('status', 'unknown')}")
            print(f"Last completed at: {marker_data.get('completion_time', 'unknown')}")
            if 'error' in marker_data:
                print(f"Last error: {marker_data['error']}")
        else:
            print("No initialization status found")
        
        # Return overall status
        return row_count > 0 and strat_exists and embeddings_available
        
    except Exception as e:
        print(f"Error checking data status: {e}")
        import traceback
        print(traceback.format_exc())
        return False


async def regenerate_derived_data(force_stratified: bool = True, force_embeddings: bool = True):
    """
    Regenerate stratified sample and/or embeddings from existing data.
    
    This is useful when data exists in the database but the stratified sample
    or embeddings need to be regenerated.
    
    Args:
        force_stratified: Whether to force regeneration of stratified sample
        force_embeddings: Whether to force regeneration of embeddings
    """
    print(f"Regenerating derived data (stratified={force_stratified}, embeddings={force_embeddings})...")
    
    # Create the process lock manager
    lock_manager = ProcessLockManager()
    
    # Try to acquire the processing lock
    if not lock_manager.acquire_lock():
        print("Another data processing instance is already running. Exiting.")
        return False
    
    try:
        # Create configuration
        config = ChanScopeConfig.from_env()
        
        # Create data manager with the config
        data_manager = ChanScopeDataManager(config)
        
        # Check if we have data in the database
        row_count = await data_manager.complete_data_storage.get_row_count()
        if row_count == 0:
            print("Error: No data found in database. Please run full data processing first.")
            return False
        
        print(f"Found {row_count} rows in database")
        
        # Generate stratified sample if requested
        if force_stratified:
            print("Regenerating stratified sample...")
            success = await data_manager.create_stratified_sample(force_refresh=True)
            if success:
                # Verify the stratified sample exists
                strat_exists = await data_manager.stratified_storage.sample_exists()
                if strat_exists:
                    sample = await data_manager.stratified_storage.get_sample()
                    if sample is not None:
                        print(f"✅ Successfully regenerated stratified sample with {len(sample)} rows")
                    else:
                        print("⚠️ Stratified sample exists but couldn't be loaded")
                else:
                    print("❌ Failed to verify stratified sample existence")
            else:
                print("❌ Failed to regenerate stratified sample")
        
        # Generate embeddings if requested
        if force_embeddings:
            print("Regenerating embeddings...")
            success = await data_manager.generate_embeddings(force_refresh=True)
            if success:
                print("✅ Successfully regenerated embeddings")
            else:
                print("❌ Failed to regenerate embeddings")
        
        # Mark initialization as complete if successful
        if (not force_stratified or success) and (not force_embeddings or success):
            lock_manager.mark_initialization_complete(True, {
                "regenerated_stratified": force_stratified,
                "regenerated_embeddings": force_embeddings
            })
        
        return True
    
    except Exception as e:
        print(f"Error regenerating derived data: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Mark initialization as failed
        lock_manager.mark_initialization_complete(False, {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "regenerated_stratified": force_stratified,
            "regenerated_embeddings": force_embeddings
        })
        
        return False
        
    finally:
        # Always release the lock when done
        lock_manager.release_lock()


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
    
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate stratified sample and embeddings from existing data"
    )
    
    parser.add_argument(
        "--stratified-only",
        action="store_true",
        help="Only regenerate stratified sample (use with --regenerate)"
    )
    
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only regenerate embeddings (use with --regenerate)"
    )
    
    parser.add_argument(
        "--ignore-lock",
        action="store_true",
        help="Ignore process locks (use with caution)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # For Replit, ensure environment is properly set
    if os.environ.get('REPLIT_ENV') or os.environ.get('REPL_ID'):
        print("Running in Replit environment")
        if not os.environ.get('REPLIT_ENV'):
            os.environ['REPLIT_ENV'] = 'replit'
    
    # Choose which function to run based on args
    if args.check:
        print("Checking data status...")
        asyncio.run(check_data_status())
    elif args.regenerate:
        regenerate_stratified = not args.embeddings_only
        regenerate_embeddings = not args.stratified_only
        print(f"Regenerating derived data (stratified={regenerate_stratified}, embeddings={regenerate_embeddings})...")
        asyncio.run(regenerate_derived_data(
            force_stratified=regenerate_stratified,
            force_embeddings=regenerate_embeddings
        ))
    else:
        print("Processing data...")
        asyncio.run(process_data(
            force_refresh=args.force_refresh,
            skip_embeddings=args.skip_embeddings
        )) 