#!/usr/bin/env python
"""
Replit Database Connectivity Check

This script checks the Replit database connectivity and validates that 
the data loading process works correctly.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("replit_db_check")

# Import environment loading utilities
from config.env_loader import load_environment
load_environment()

def check_replit_env():
    """Check if we're in a Replit environment."""
    is_replit = os.environ.get('REPLIT_ENV', '').lower() in ('replit', 'true') or os.environ.get('REPL_ID') is not None
    logger.info(f"Replit environment detected: {is_replit}")
    
    if not is_replit:
        logger.warning("This script is intended to be run in a Replit environment")
    
    return is_replit

def check_postgres_connection():
    """Check PostgreSQL database connection."""
    logger.info("Checking PostgreSQL connection...")
    
    # Import the PostgresDB class
    from config.replit import PostgresDB
    
    try:
        # Initialize the PostgresDB class
        db = PostgresDB()
        
        # Test connection with a simple query
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
        logger.info("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False

def check_database_schema():
    """Check if database schema exists and is properly initialized."""
    logger.info("Checking database schema...")
    
    from config.replit import PostgresDB
    
    try:
        db = PostgresDB()
        
        # Check for required tables
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if complete_data table exists
            cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'complete_data'
            )
            """)
            complete_data_exists = cursor.fetchone()[0]
            
            # Check if metadata table exists
            cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'metadata'
            )
            """)
            metadata_exists = cursor.fetchone()[0]
            
        if complete_data_exists and metadata_exists:
            logger.info("✅ Database schema is properly initialized")
            
            # Get row count to provide more info
            row_count = db.get_row_count()
            logger.info(f"   - complete_data table has {row_count} rows")
            
            # Check last updated timestamp
            last_update = db.get_data_updated_timestamp()
            if last_update:
                logger.info(f"   - Data was last updated at: {last_update}")
            else:
                logger.info("   - No update timestamp found")
            
            return True
        else:
            missing = []
            if not complete_data_exists:
                missing.append("complete_data")
            if not metadata_exists:
                missing.append("metadata")
            
            logger.error(f"❌ Missing tables: {', '.join(missing)}")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking database schema: {e}")
        return False

def check_keyvalue_store():
    """Check if the Replit Key-Value store is accessible."""
    logger.info("Checking Replit Key-Value store...")
    
    try:
        from replit import db as kv_db
        
        # Test key presence
        test_key = f"test_key_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        kv_db[test_key] = "test_value"
        
        # Verify key was stored
        if kv_db[test_key] == "test_value":
            logger.info("✅ Replit Key-Value store is working")
            
            # Clean up test key
            del kv_db[test_key]
            return True
        else:
            logger.error("❌ Key-Value store test failed: value mismatch")
            return False
    except Exception as e:
        logger.error(f"❌ Key-Value store test failed: {e}")
        return False

def check_object_storage():
    """Check if Replit Object Storage is accessible."""
    logger.info("Checking Replit Object Storage...")
    
    try:
        # Try to import Object Storage
        try:
            from replit.object_storage import Client
        except ImportError:
            logger.error("❌ replit-object-storage package not installed, run 'pip install replit-object-storage'")
            return False
            
        # Initialize client
        client = Client()
        
        # Create a test file
        test_key = f"test_object_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        test_content = "This is a test object for embedding storage"
        
        # Upload to Object Storage
        client.upload_from_text(test_key, test_content)
        
        # Verify content was stored correctly
        retrieved_content = client.download_as_text(test_key)
        
        if retrieved_content == test_content:
            logger.info("✅ Replit Object Storage is working")
            
            # Clean up test object
            client.delete(test_key)
            return True
        else:
            logger.error("❌ Object Storage test failed: content mismatch")
            return False
    except Exception as e:
        logger.error(f"❌ Object Storage test failed: {e}")
        return False

def test_data_loading():
    """Test the data loading process."""
    logger.info("Testing data loading process...")
    
    try:
        from config.replit import PostgresDB
        db = PostgresDB()
        
        # Create a small test dataframe
        test_data = {
            'thread_id': [f'test_{i}' for i in range(3)],
            'content': ['Test content 1', 'Test content 2', 'Test content 3'],
            'posted_date_time': [pd.Timestamp.now() for _ in range(3)]
        }
        df = pd.DataFrame(test_data)
        
        # Test sync_data_from_dataframe method
        rows_added = db.sync_data_from_dataframe(df)
        logger.info(f"   - Sync method added {rows_added} rows")
        
        # Check if data can be retrieved
        test_df = db.get_complete_data()
        test_rows = test_df[test_df['thread_id'].str.startswith('test_')].shape[0]
        logger.info(f"   - Retrieved {test_rows} test rows")
        
        if test_rows >= rows_added:
            logger.info("✅ Data loading process is working")
            return True
        else:
            logger.error("❌ Data loading test failed: retrieved fewer rows than added")
            return False
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all checks."""
    logger.info("Starting Replit database connectivity check...")
    
    # Check if we're in a Replit environment
    is_replit = check_replit_env()
    if not is_replit:
        logger.warning("Not in Replit environment, but continuing with checks...")
    
    # Run all checks
    checks = [
        ("PostgreSQL Connection", check_postgres_connection()),
        ("Database Schema", check_database_schema()),
        ("Key-Value Store", check_keyvalue_store()),
        ("Object Storage", check_object_storage()),
        ("Data Loading Process", test_data_loading())
    ]
    
    # Print summary
    logger.info("\n--- Check Summary ---")
    all_passed = True
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
        all_passed = all_passed and result
    
    if all_passed:
        logger.info("\n✅ All checks passed! The Replit database integration is working correctly.")
        return 0
    else:
        logger.error("\n❌ Some checks failed. Please review the logs and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 