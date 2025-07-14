#!/usr/bin/env python
"""
Wipe All Data Script for Replit Environment

This script completely clears all data from both PostgreSQL database
and Replit Key-Value store to allow for a fresh start.

WARNING: This operation is irreversible. All data will be permanently lost.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_wipe")

# Import environment loading utilities
from config.env_loader import load_environment
load_environment()

def confirm_wipe():
    """Ask for user confirmation before proceeding with data wipe."""
    print("\n" + "="*60)
    print("WARNING: DESTRUCTIVE OPERATION")
    print("="*60)
    print("This will permanently delete ALL data including:")
    print("- Complete dataset in PostgreSQL")
    print("- All metadata records")
    print("- Stratified samples in Key-Value store")
    print("- All embeddings and thread mappings")
    print("- Processing state information")
    print("- Any cached or temporary data")
    print("\nThis operation CANNOT be undone!")
    print("="*60)
    
    # Require typing "WIPE ALL DATA" to confirm
    confirmation = input("\nType 'WIPE ALL DATA' to confirm (case sensitive): ")
    
    if confirmation != "WIPE ALL DATA":
        print("Operation cancelled.")
        return False
    
    # Double confirmation
    final_confirm = input("\nAre you absolutely sure? Type 'YES' to proceed: ")
    if final_confirm != "YES":
        print("Operation cancelled.")
        return False
    
    return True

def wipe_postgres_data():
    """Wipe all data from PostgreSQL database."""
    logger.info("Starting PostgreSQL data wipe...")
    
    try:
        from config.replit import PostgresDB
        db = PostgresDB()
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get initial row counts for logging
            cursor.execute("SELECT COUNT(*) FROM complete_data")
            complete_data_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM metadata")
            metadata_count = cursor.fetchone()[0]
            
            logger.info(f"Found {complete_data_count} rows in complete_data table")
            logger.info(f"Found {metadata_count} rows in metadata table")
            
            # Truncate tables (faster than DELETE and resets auto-increment)
            logger.info("Truncating complete_data table...")
            cursor.execute("TRUNCATE TABLE complete_data RESTART IDENTITY CASCADE")
            
            logger.info("Truncating metadata table...")
            cursor.execute("TRUNCATE TABLE metadata RESTART IDENTITY CASCADE")
            
            # Commit the changes
            conn.commit()
            
            # Verify tables are empty
            cursor.execute("SELECT COUNT(*) FROM complete_data")
            remaining_complete = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM metadata")
            remaining_metadata = cursor.fetchone()[0]
            
            if remaining_complete == 0 and remaining_metadata == 0:
                logger.info("✅ PostgreSQL data wipe completed successfully")
                return True
            else:
                logger.error(f"❌ Wipe incomplete: {remaining_complete} rows remain in complete_data, {remaining_metadata} in metadata")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error wiping PostgreSQL data: {e}")
        return False

def wipe_keyvalue_store():
    """Wipe all data from Replit Key-Value store."""
    logger.info("Starting Key-Value store data wipe...")
    
    try:
        from replit import db as kv_db
        from config.replit import KeyValueStore
        
        # Get all keys to see what we're working with
        all_keys = list(kv_db.keys())
        logger.info(f"Found {len(all_keys)} keys in Key-Value store")
        
        if all_keys:
            logger.info(f"Keys to be deleted: {all_keys}")
        
        # Initialize KeyValueStore to get key constants
        kv_store = KeyValueStore()
        
        # Track keys we'll delete
        deleted_keys = []
        
        # Delete specific known keys
        known_keys = [
            kv_store.STRATIFIED_SAMPLE_KEY,
            kv_store.EMBEDDINGS_KEY,
            kv_store.THREAD_MAP_KEY,
            kv_store.STATE_KEY,
            f"{kv_store.EMBEDDINGS_KEY}_meta"
        ]
        
        for key in known_keys:
            if key in kv_db:
                del kv_db[key]
                deleted_keys.append(key)
                logger.info(f"Deleted key: {key}")
        
        # Delete any embeddings chunks (pattern: embeddings_chunk_N)
        for key in all_keys:
            if key.startswith(f"{kv_store.EMBEDDINGS_KEY}_chunk_"):
                del kv_db[key]
                deleted_keys.append(key)
                logger.info(f"Deleted embeddings chunk: {key}")
        
        # Delete any other keys that might be related to our application
        # (be careful not to delete unrelated keys if sharing the KV store)
        app_related_patterns = [
            "stratified_",
            "embeddings_",
            "thread_",
            "processing_",
            "chanscope_",
            "data_"
        ]
        
        for key in all_keys:
            if key not in deleted_keys:  # Don't double-delete
                for pattern in app_related_patterns:
                    if key.startswith(pattern):
                        del kv_db[key]
                        deleted_keys.append(key)
                        logger.info(f"Deleted app-related key: {key}")
                        break
        
        # Final verification
        remaining_keys = list(kv_db.keys())
        logger.info(f"Deleted {len(deleted_keys)} keys from Key-Value store")
        
        if remaining_keys:
            logger.info(f"Remaining keys (likely unrelated to our app): {remaining_keys}")
        
        logger.info("✅ Key-Value store wipe completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error wiping Key-Value store: {e}")
        return False

def wipe_local_cache_files():
    """Remove any local cache files and temporary data."""
    logger.info("Cleaning up local cache files...")
    
    try:
        # Paths to clean up
        paths_to_clean = [
            "data/stratified",
            "data/.replit_init_complete",
            "data/.scheduler_pid",
            "temp_files",
            "logs"
        ]
        
        cleaned_count = 0
        
        for path_str in paths_to_clean:
            path = Path(path_str)
            
            if path.is_file():
                path.unlink()
                logger.info(f"Removed file: {path}")
                cleaned_count += 1
            elif path.is_dir():
                # Remove all files in directory but keep the directory
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                logger.info(f"Cleaned directory: {path}")
        
        logger.info(f"✅ Cleaned up {cleaned_count} local files")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cleaning local cache files: {e}")
        return False

def main():
    """Main function to orchestrate the complete data wipe."""
    logger.info("Starting complete data wipe process...")
    
    # Require confirmation
    if not confirm_wipe():
        logger.info("Data wipe cancelled by user.")
        return 0
    
    logger.info("Proceeding with data wipe...")
    
    # Track success of each operation
    operations = []
    
    # 1. Wipe PostgreSQL data
    postgres_success = wipe_postgres_data()
    operations.append(("PostgreSQL Data", postgres_success))
    
    # 2. Wipe Key-Value store
    kv_success = wipe_keyvalue_store()
    operations.append(("Key-Value Store", kv_success))
    
    # 3. Clean local cache files
    cache_success = wipe_local_cache_files()
    operations.append(("Local Cache", cache_success))
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("DATA WIPE SUMMARY")
    logger.info("="*50)
    
    all_successful = True
    for operation_name, success in operations:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status}: {operation_name}")
        all_successful = all_successful and success
    
    if all_successful:
        logger.info("\n✅ ALL DATA WIPED SUCCESSFULLY!")
        logger.info("The system is now in a clean state and ready for fresh data.")
        logger.info("You can now run the initialization process again.")
        return 0
    else:
        logger.error("\n❌ SOME OPERATIONS FAILED!")
        logger.error("Please review the logs and manually address any remaining data.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 