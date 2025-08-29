#!/usr/bin/env python
"""
Wipe All Data Script (Dev + Prod)

Safely clears all persisted data used by the app in both development and
production setups. Supports:
  - Replit PostgreSQL (dev or a supplied production DSN)
  - Replit Key-Value store (dev)
  - Replit Object Storage (stratified sample, embeddings, query results)
  - File-based storage (Docker/local): complete_data.csv, stratified, embeddings,
    generated_data, temp files, logs

WARNING: This operation is irreversible. All data will be permanently lost.
"""

import os
import sys
import logging
from pathlib import Path
import argparse
import contextlib
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
from config.env_loader import load_environment, is_replit_environment, is_docker_environment
load_environment()

def confirm_wipe(non_interactive: bool = False) -> bool:
    """Ask for user confirmation before proceeding with data wipe."""
    if non_interactive:
        # Explicitly bypass interactive confirmation when --yes is provided
        logger.warning("Bypassing interactive confirmation due to --yes flag")
        return True
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

@contextlib.contextmanager
def _env_overrides(**pairs):
    """Temporarily set environment variables within a context manager."""
    saved = {}
    try:
        for k, v in pairs.items():
            saved[k] = os.environ.get(k)
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old

def wipe_postgres_data(database_url: str = None, pg_host: str = None, pg_user: str = None, pg_password: str = None) -> bool:
    """Wipe all data from PostgreSQL database."""
    logger.info("Starting PostgreSQL data wipe...")
    # Allow overriding connection details to target production DB explicitly
    try:
        overrides = {}
        if database_url:
            overrides["DATABASE_URL"] = database_url
        if pg_host:
            overrides["PGHOST"] = pg_host
        if pg_user:
            overrides["PGUSER"] = pg_user
        if pg_password:
            overrides["PGPASSWORD"] = pg_password

        from config.replit import PostgresDB

        with _env_overrides(**overrides):
            db = PostgresDB()
            
            # Display target for clarity (mask password)
            target = os.environ.get("DATABASE_URL") or f"{os.environ.get('PGUSER','?')}@{os.environ.get('PGHOST','?')}"
            logger.info(f"Targeting PostgreSQL: {target}")

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
                    logger.error(
                        f"❌ Wipe incomplete: {remaining_complete} rows remain in complete_data, {remaining_metadata} in metadata"
                    )
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
        from config.settings import Config
        paths = Config.get_paths()
        # Common files and directories to clear
        paths_to_clean = [
            # CSV complete dataset
            Path(paths.get('root_data_path', 'data')) / 'complete_data.csv',
            # Stratified + embeddings
            Path(paths.get('stratified', 'data/stratified')),
            Path(paths.get('stratified', 'data/stratified')) / 'embeddings.npz',
            Path(paths.get('stratified', 'data/stratified')) / 'thread_id_map.json',
            # Generated outputs (query results, etc.)
            Path(paths.get('generated_data', 'data/generated_data')),
            # Temp + logs + internal flags
            Path(paths.get('temp', 'temp_files')),
            Path(paths.get('logs', 'logs')),
            Path(paths.get('root_data_path', 'data')) / '.replit_init_complete',
            Path(paths.get('root_data_path', 'data')) / '.scheduler_pid',
        ]
        
        cleaned_count = 0
        
        for path_str in paths_to_clean:
            path = Path(path_str)
            
            if path.is_file():
                path.unlink()
                logger.info(f"Removed file: {path}")
                cleaned_count += 1
            elif path.is_dir():
                # Remove all files in directory but keep the directory structure
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

def wipe_object_storage(delete_query_artifacts: bool = True) -> bool:
    """Remove artifacts stored in Replit Object Storage.

    Deletes:
      - Stratified sample and its metadata
      - Embeddings and thread_id_map
      - Query results under prefix query_results/ (optional)
    """
    logger.info("Starting Object Storage wipe...")
    try:
        from config.chanscope_config import ChanScopeConfig
        from config.storage import ReplitStratifiedSampleStorage, ReplitObjectEmbeddingStorage

        cfg = ChanScopeConfig.from_env()

        # Stratified + query results bucket (default bucket)
        strat = ReplitStratifiedSampleStorage(cfg)
        strat_client = strat._init_object_client()
        deleted = 0
        if strat_client:
            try:
                objects = strat_client.list()
                for obj in objects:
                    name = getattr(obj, 'name', str(obj))
                    if name in {strat.stratified_key, strat.metadata_key}:
                        try:
                            if hasattr(strat_client, 'delete'):
                                strat_client.delete(name)
                            elif hasattr(strat_client, 'remove'):
                                strat_client.remove(name)
                            deleted += 1
                            logger.info(f"Deleted object: {name}")
                        except Exception as del_err:
                            logger.warning(f"Failed to delete {name}: {del_err}")
                    elif delete_query_artifacts and (name.startswith('query_results/') or name.startswith('embeddings/')):
                        try:
                            if hasattr(strat_client, 'delete'):
                                strat_client.delete(name)
                            elif hasattr(strat_client, 'remove'):
                                strat_client.remove(name)
                            deleted += 1
                            logger.info(f"Deleted artifact: {name}")
                        except Exception as del_err:
                            logger.warning(f"Failed to delete artifact {name}: {del_err}")
            except Exception as list_err:
                logger.warning(f"Failed to list default bucket objects: {list_err}")
        else:
            logger.warning("No Object Storage client available for stratified bucket")

        # Embeddings bucket
        emb = ReplitObjectEmbeddingStorage(cfg)
        emb_client = emb._init_object_client()
        if emb_client:
            for name in [emb.embeddings_key, emb.thread_map_key]:
                try:
                    if hasattr(emb_client, 'delete'):
                        emb_client.delete(name)
                    elif hasattr(emb_client, 'remove'):
                        emb_client.remove(name)
                    deleted += 1
                    logger.info(f"Deleted embedding object: {name}")
                except Exception as del_err:
                    logger.warning(f"Failed to delete embedding object {name}: {del_err}")
        else:
            logger.warning("No Object Storage client available for embeddings bucket")

        logger.info(f"✅ Object Storage wipe completed. Deleted {deleted} objects")
        return True
    except Exception as e:
        logger.error(f"❌ Error wiping Object Storage: {e}")
        return False

def main():
    """Main function to orchestrate the complete data wipe."""
    parser = argparse.ArgumentParser(description="Wipe all persisted app data (dev + prod capable)")
    parser.add_argument('--yes', '-y', action='store_true', help='Skip interactive confirmations')
    # Postgres overrides for targeting production DB explicitly
    parser.add_argument('--database-url', help='Override DATABASE_URL to target a specific PostgreSQL instance')
    parser.add_argument('--pg-host', help='Override PGHOST (used if DATABASE_URL not provided)')
    parser.add_argument('--pg-user', help='Override PGUSER')
    parser.add_argument('--pg-password', help='Override PGPASSWORD')
    # Scope controls
    parser.add_argument('--no-kv', action='store_true', help='Skip wiping Replit Key-Value store')
    parser.add_argument('--no-objects', action='store_true', help='Skip wiping Replit Object Storage')
    parser.add_argument('--no-files', action='store_true', help='Skip wiping file-based artifacts')

    args = parser.parse_args()

    logger.info("Starting complete data wipe process...")

    # Require confirmation
    if not confirm_wipe(non_interactive=args.yes):
        logger.info("Data wipe cancelled by user.")
        return 0
    
    logger.info("Proceeding with data wipe...")
    
    # Track success of each operation
    operations = []
    
    # 1. Wipe PostgreSQL data (works for both dev + prod if overrides provided)
    postgres_success = wipe_postgres_data(
        database_url=args.database_url,
        pg_host=args.pg_host,
        pg_user=args.pg_user,
        pg_password=args.pg_password,
    )
    operations.append(("PostgreSQL Data", postgres_success))
    
    # 2. Wipe Key-Value store (Replit only)
    kv_success = True
    if not args.no_kv and is_replit_environment():
        kv_success = wipe_keyvalue_store()
        operations.append(("Key-Value Store", kv_success))
    else:
        operations.append(("Key-Value Store (skipped)", kv_success))
    
    # 3. Object Storage wipe (Replit only)
    obj_success = True
    if not args.no_objects and is_replit_environment():
        obj_success = wipe_object_storage(delete_query_artifacts=True)
        operations.append(("Object Storage", obj_success))
    else:
        operations.append(("Object Storage (skipped)", obj_success))

    # 4. Clean local/file-based artifacts (Docker/local and also dev workspace outputs)
    files_success = True
    if not args.no_files:
        files_success = wipe_local_cache_files()
    operations.append(("Local/File Artifacts", files_success))
    
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
