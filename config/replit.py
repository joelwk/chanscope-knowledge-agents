"""
Replit database integration for Chanscope.

This module implements the Chanscope data architecture using Replit's
PostgreSQL database and Key-Value store services.
"""

import os
import json
import time
import pickle
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import sql
from pathlib import Path
from replit import db as kv_db
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Awaitable
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Exception raised for database connection errors."""
    pass

class PostgresDB:
    """PostgreSQL database handler for complete dataset storage and management."""
    
    def __init__(self):
        """Initialize PostgreSQL connection parameters from environment variables."""
        # Replit automatically sets these environment variables when you add a PostgreSQL database
        self.database_url = os.environ.get('DATABASE_URL', '')
        self.pghost = os.environ.get('PGHOST', '')
        self.pguser = os.environ.get('PGUSER', '')
        self.pgpassword = os.environ.get('PGPASSWORD', '')
        
        # For KeyValue store (separate from PostgreSQL)
        self.kv_connection_url = os.environ.get('REPLIT_DB_URL', '')
        
        # Log connection availability (not the credentials themselves)
        if self.database_url:
            logger.info("PostgreSQL DATABASE_URL is available")
        else:
            logger.warning("PostgreSQL DATABASE_URL is not set")
        
        if all([self.pghost, self.pguser, self.pgpassword]):
            logger.info("PostgreSQL individual connection parameters are available")
        else:
            logger.warning("Some PostgreSQL connection parameters are missing")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures connections are properly closed after use.
        """
        connection = None
        try:
            # Prefer using the full connection string if available
            if self.database_url:
                connection = psycopg2.connect(self.database_url)
            else:
                # Fall back to individual parameters
                connection = psycopg2.connect(
                    host=self.pghost,
                    user=self.pguser,
                    password=self.pgpassword
                )
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}")
        finally:
            if connection:
                connection.close()
    
    def initialize_schema(self):
        """Create the database schema if it doesn't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Create table to store the complete dataset
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS complete_data (
                    id SERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    content TEXT,
                    posted_date_time TIMESTAMP WITH TIME ZONE,
                    channel_name TEXT,
                    author TEXT,
                    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create index on timestamp for efficient time-based queries
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posted_date_time 
                ON complete_data (posted_date_time);
                """)
                
                # Create index on thread_id for efficient lookups
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_id 
                ON complete_data (thread_id);
                """)
                
                # Create metadata table to track processing state
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id SERIAL PRIMARY KEY,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error initializing schema: {e}")
                raise
    
    def insert_complete_data(self, df: pd.DataFrame):
        """
        Insert data into the complete_data table.
        
        Args:
            df: DataFrame containing data to insert
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to insert")
            return
        
        # Log the incoming dataframe columns and shape
        logger.info(f"Inserting data with shape {df.shape} and columns {list(df.columns)}")
        
        # Ensure expected columns exist
        required_columns = ['thread_id', 'content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to automatically map columns
            for missing_col in missing_columns:
                if missing_col == 'content' and any(col in df.columns for col in ['text', 'text_clean', 'message']):
                    # Map a text-like column to content
                    for possible_col in ['text', 'text_clean', 'message']:
                        if possible_col in df.columns:
                            df['content'] = df[possible_col]
                            logger.info(f"Mapped {possible_col} to content column")
                            break
                elif missing_col == 'thread_id' and any(col in df.columns for col in ['id', 'message_id', 'post_id']):
                    # Map an ID-like column to thread_id
                    for possible_col in ['id', 'message_id', 'post_id']:
                        if possible_col in df.columns:
                            df['thread_id'] = df[possible_col]
                            logger.info(f"Mapped {possible_col} to thread_id column")
                            break
            
            # Check if we still have missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Ensure posted_date_time is present or create it
        if 'posted_date_time' not in df.columns:
            if 'date' in df.columns:
                df['posted_date_time'] = df['date']
                logger.info("Mapped date to posted_date_time column")
            elif 'timestamp' in df.columns:
                df['posted_date_time'] = df['timestamp']
                logger.info("Mapped timestamp to posted_date_time column")
            else:
                # Create a default timestamp
                df['posted_date_time'] = pd.Timestamp.now()
                logger.info("Created default posted_date_time column with current time")
        
        # Format posted_date_time as timestamp if it's not already
        if not pd.api.types.is_datetime64_dtype(df['posted_date_time']):
            df['posted_date_time'] = pd.to_datetime(df['posted_date_time'], errors='coerce')
            # Drop rows with invalid dates
            invalid_dates = df['posted_date_time'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Dropping {invalid_dates} rows with invalid dates")
                df = df.dropna(subset=['posted_date_time'])
        
        # Make sure thread_id is string
        df['thread_id'] = df['thread_id'].astype(str)
        
        # Add optional columns if not present
        if 'channel_name' not in df.columns:
            df['channel_name'] = 'default'
        
        if 'author' not in df.columns:
            df['author'] = 'unknown'
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Use execute_values for faster bulk inserts
                from psycopg2.extras import execute_values
                
                # Prepare data as list of tuples
                data = [
                    (row['thread_id'], row['content'], row['posted_date_time'], 
                     row['channel_name'], row['author'])
                    for _, row in df.iterrows()
                ]
                
                execute_values(
                    cursor,
                    """
                    INSERT INTO complete_data 
                    (thread_id, content, posted_date_time, channel_name, author)
                    VALUES %s
                    """,
                    data
                )
                
                conn.commit()
                logger.info(f"Inserted {len(df)} rows into complete_data table")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    
    def get_complete_data(self, filter_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve complete dataset from the database.
        
        Args:
            filter_date: Optional date string to filter data from
            
        Returns:
            DataFrame containing the complete dataset
        """
        query = "SELECT * FROM complete_data"
        params = []
        
        if filter_date:
            query += " WHERE posted_date_time >= %s"
            params.append(filter_date)
            
        query += " ORDER BY posted_date_time DESC"
        
        with self.get_connection() as conn:
            try:
                return pd.read_sql_query(query, conn, params=params)
            except Exception as e:
                logger.error(f"Error retrieving complete data: {e}")
                raise
    
    def update_metadata(self, key: str, value: Any):
        """
        Update or insert a metadata record.
        
        Args:
            key: Metadata key
            value: Metadata value (will be converted to JSON string)
        """
        json_value = json.dumps(value)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Upsert operation - update if exists, insert if not
                cursor.execute("""
                INSERT INTO metadata (key, value, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (key) 
                DO UPDATE SET 
                    value = EXCLUDED.value,
                    updated_at = CURRENT_TIMESTAMP
                """, (key, json_value))
                
                conn.commit()
                logger.debug(f"Updated metadata: {key}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error updating metadata: {e}")
                raise
    
    def get_metadata(self, key: str) -> Optional[Any]:
        """
        Retrieve a metadata value.
        
        Args:
            key: Metadata key to retrieve
            
        Returns:
            Parsed value or None if key doesn't exist
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT value FROM metadata WHERE key = %s", (key,))
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result[0])
                return None
            except Exception as e:
                logger.error(f"Error retrieving metadata: {e}")
                raise
    
    def get_data_updated_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp when the data was last updated.
        
        Returns:
            Datetime of last update or None if not available
        """
        timestamp = self.get_metadata('last_data_update')
        if timestamp:
            return datetime.fromisoformat(timestamp)
        return None
    
    def get_row_count(self) -> int:
        """
        Get the count of rows in the complete_data table.
        
        Returns:
            Integer count of rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM complete_data")
                result = cursor.fetchone()
                return result[0] if result else 0
            except Exception as e:
                logger.error(f"Error getting row count: {e}")
                raise
    
    def perform_stratified_sampling(self, sample_size: int, time_column: str) -> pd.DataFrame:
        """
        Perform stratified sampling directly in the database.
        
        This implements the same logic as sampler.py's sample_by_time but using SQL.
        
        Args:
            sample_size: Target sample size
            time_column: Name of the time column for stratification
            
        Returns:
            DataFrame containing the stratified sample
        """
        with self.get_connection() as conn:
            try:
                # Calculate number of time periods (days)
                cursor = conn.cursor()
                cursor.execute("""
                SELECT 
                    COUNT(DISTINCT DATE_TRUNC('day', posted_date_time)) 
                FROM complete_data
                WHERE posted_date_time IS NOT NULL
                """)
                n_periods = cursor.fetchone()[0]
                
                if n_periods == 0:
                    logger.warning("No time periods found in data")
                    return pd.DataFrame()
                
                # Calculate samples per period
                samples_per_period = max(5, sample_size // n_periods)
                
                # Execute stratified sampling query
                query = """
                WITH time_groups AS (
                    SELECT 
                        *,
                        DATE_TRUNC('day', posted_date_time) AS day_group
                    FROM complete_data
                    WHERE posted_date_time IS NOT NULL
                ),
                samples AS (
                    SELECT *
                    FROM (
                        SELECT 
                            *,
                            ROW_NUMBER() OVER (
                                PARTITION BY day_group 
                                ORDER BY RANDOM()
                            ) as row_num
                        FROM time_groups
                    ) ranked
                    WHERE row_num <= %s
                )
                SELECT 
                    id, thread_id, content, posted_date_time, channel_name, author
                FROM samples
                ORDER BY RANDOM()
                LIMIT %s
                """
                
                # Execute the query and return results
                return pd.read_sql_query(query, conn, params=(samples_per_period, sample_size))
            except Exception as e:
                logger.error(f"Error performing stratified sampling: {e}")
                raise

    def check_data_needs_update(self, since_date: Optional[str] = None) -> Tuple[bool, Optional[datetime]]:
        """
        Check if the database needs to be updated with new data.
        
        Args:
            since_date: Optional date string to check data from
            
        Returns:
            Tuple of (needs_update, last_updated_timestamp)
        """
        last_updated = self.get_data_updated_timestamp()
        
        if last_updated is None:
            # No data exists, definitely needs update
            return True, None
        
        # If since_date is provided, check if our data is older than that
        if since_date:
            try:
                target_date = datetime.fromisoformat(since_date.replace('Z', '+00:00'))
                if last_updated < target_date:
                    return True, last_updated
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing since_date '{since_date}': {e}")
        
        # Check if data is more than 1 day old
        if (datetime.now(last_updated.tzinfo) - last_updated) > timedelta(days=1):
            return True, last_updated
        
        return False, last_updated

    def sync_data_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Synchronize data from DataFrame to database.
        Only inserts rows that don't already exist.
        
        Args:
            df: DataFrame containing data to sync
            
        Returns:
            Number of new rows inserted
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to sync")
            return 0
        
        # Log incoming data information
        logger.info(f"Syncing data with shape {df.shape} and columns {list(df.columns)}")
        
        # Ensure expected columns exist with same mapping logic as insert_complete_data
        required_columns = ['thread_id', 'content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to automatically map columns
            for missing_col in missing_columns:
                if missing_col == 'content' and any(col in df.columns for col in ['text', 'text_clean', 'message']):
                    # Map a text-like column to content
                    for possible_col in ['text', 'text_clean', 'message']:
                        if possible_col in df.columns:
                            df['content'] = df[possible_col]
                            logger.info(f"Mapped {possible_col} to content column")
                            break
                elif missing_col == 'thread_id' and any(col in df.columns for col in ['id', 'message_id', 'post_id']):
                    # Map an ID-like column to thread_id
                    for possible_col in ['id', 'message_id', 'post_id']:
                        if possible_col in df.columns:
                            df['thread_id'] = df[possible_col]
                            logger.info(f"Mapped {possible_col} to thread_id column")
                            break
            
            # Check if we still have missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Ensure posted_date_time is present or create it
        if 'posted_date_time' not in df.columns:
            if 'date' in df.columns:
                df['posted_date_time'] = df['date']
                logger.info("Mapped date to posted_date_time column")
            elif 'timestamp' in df.columns:
                df['posted_date_time'] = df['timestamp']
                logger.info("Mapped timestamp to posted_date_time column")
            else:
                # Create a default timestamp
                df['posted_date_time'] = pd.Timestamp.now()
                logger.info("Created default posted_date_time column with current time")
        
        # Format posted_date_time as timestamp if it's not already
        if not pd.api.types.is_datetime64_dtype(df['posted_date_time']):
            df['posted_date_time'] = pd.to_datetime(df['posted_date_time'], errors='coerce')
            # Drop rows with invalid dates
            invalid_dates = df['posted_date_time'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Dropping {invalid_dates} rows with invalid dates")
                df = df.dropna(subset=['posted_date_time'])
        
        # Make sure thread_id is string
        df['thread_id'] = df['thread_id'].astype(str)
        
        # Add optional columns if not present
        if 'channel_name' not in df.columns:
            df['channel_name'] = 'default'
        
        if 'author' not in df.columns:
            df['author'] = 'unknown'
        
        # Get existing thread_ids
        existing_thread_ids = set()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT DISTINCT thread_id FROM complete_data")
                for row in cursor.fetchall():
                    existing_thread_ids.add(row[0])
            except Exception as e:
                logger.error(f"Error retrieving existing thread_ids: {e}")
                raise
        
        # Filter to only new rows
        new_rows_df = df[~df['thread_id'].isin(existing_thread_ids)]
        
        if new_rows_df.empty:
            logger.info("No new rows to insert")
            return 0
        
        # Insert new rows
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Use execute_values for faster bulk inserts
                from psycopg2.extras import execute_values
                
                # Columns we will insert
                columns = ['thread_id', 'content', 'posted_date_time', 'channel_name', 'author']
                
                # Build data tuples with all columns
                data = []
                for _, row in new_rows_df.iterrows():
                    row_data = [
                        row['thread_id'], 
                        row['content'], 
                        row['posted_date_time'],
                        row.get('channel_name', 'default'),
                        row.get('author', 'unknown')
                    ]
                    data.append(tuple(row_data))
                
                # Build SQL statement
                sql = f"""
                INSERT INTO complete_data 
                (thread_id, content, posted_date_time, channel_name, author)
                VALUES %s
                """
                
                execute_values(cursor, sql, data)
                
                # Update the last_updated metadata
                self.update_metadata('data_last_updated', datetime.now().isoformat())
                
                conn.commit()
                logger.info(f"Inserted {len(new_rows_df)} new rows into complete_data table")
                return len(new_rows_df)
            except Exception as e:
                conn.rollback()
                logger.error(f"Error inserting data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise


class KeyValueStore:
    """Key-Value store handler for stratified samples and embeddings."""
    
    # Key prefixes for different data types
    STRATIFIED_SAMPLE_KEY = "stratified_sample"
    EMBEDDINGS_KEY = "embeddings"
    THREAD_MAP_KEY = "thread_id_map"
    STATE_KEY = "processing_state"
    
    def __init__(self):
        """Initialize the Key-Value store connection."""
        # Check if REPLIT_DB_URL is set
        replit_db_url = os.environ.get('REPLIT_DB_URL', '')
        if not replit_db_url:
            # For deployments, check the tmp file location
            try:
                if os.path.exists('/tmp/replitdb'):
                    with open('/tmp/replitdb', 'r') as f:
                        replit_db_url = f.read().strip()
                    logger.info("Retrieved REPLIT_DB_URL from /tmp/replitdb for deployment")
                    os.environ['REPLIT_DB_URL'] = replit_db_url
            except Exception as e:
                logger.warning(f"Could not read from /tmp/replitdb: {e}")
        
        if replit_db_url:
            logger.info("REPLIT_DB_URL is available for Key-Value store operations")
        else:
            logger.warning("REPLIT_DB_URL is not set, Key-Value operations may fail")
            
        # Import the db here to ensure it uses the environment variable we may have just set
        from replit import db
        self.db = db
    
    def store_stratified_sample(self, df: pd.DataFrame) -> bool:
        """
        Store stratified sample in the Replit Key-Value store.
        
        Args:
            df: DataFrame containing the stratified sample
            
        Returns:
            bool: True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return False
            
        try:
            # Create a deep copy to avoid modifying the original
            df_copy = df.copy()
            
            # Convert datetime columns to ISO format strings to avoid serialization issues
            for col in df_copy.select_dtypes(include=['datetime64']).columns:
                df_copy[col] = df_copy[col].astype(str)
            
            # Convert to dict with a simpler structure to avoid circular references
            data_records = []
            for _, row in df_copy.iterrows():
                # Convert each row to a dict with simple types only
                row_dict = {}
                for col in df_copy.columns:
                    # Handle special types that might cause circular references
                    val = row[col]
                    if isinstance(val, (np.ndarray, list)):
                        # Skip embedding arrays for now - they'll be stored separately
                        continue
                    elif isinstance(val, (int, float, str, bool)) or val is None:
                        # These types are safe for JSON
                        row_dict[col] = val
                    else:
                        # Convert anything else to string
                        row_dict[col] = str(val)
                data_records.append(row_dict)
            
            # Build a simpler dict structure
            df_dict = {
                'data': data_records,
                'columns': list(df_copy.columns),
                'row_count': len(df_copy)
            }
            
            # Try JSON serialization first
            try:
                json_data = json.dumps(df_dict)
                # Store in Replit DB
                from replit import db as kv_db
                kv_db[self.STRATIFIED_SAMPLE_KEY] = json_data
                logger.info(f"Stored stratified sample with {len(df)} rows as JSON in key-value store")
                return True
            except TypeError as json_err:
                logger.warning(f"JSON serialization failed, falling back to pickle: {json_err}")
                
                # If JSON fails, try pickle with protocol 4 (more efficient)
                try:
                    # Use pickle protocol 4 for better handling of large objects
                    pickle_data = pickle.dumps(df_dict, protocol=4)
                    from replit import db as kv_db
                    kv_db[self.STRATIFIED_SAMPLE_KEY] = pickle_data
                    logger.info(f"Stored stratified sample with {len(df)} rows as pickle in key-value store")
                    return True
                except Exception as pickle_err:
                    logger.error(f"Pickle serialization also failed: {pickle_err}")
                    return False
                
        except Exception as e:
            logger.error(f"Error storing stratified sample: {e}")
            return False
            
    def get_stratified_sample(self) -> pd.DataFrame:
        """
        Get stratified sample from the Replit Key-Value store.
        
        Returns:
            DataFrame containing the stratified sample or None if not found
        """
        try:
            # Get replit db module
            from replit import db as kv_db
            
            if self.STRATIFIED_SAMPLE_KEY not in kv_db:
                logger.info("No stratified sample found in key-value store")
                return None
                
            # Get the stored value
            stored_value = kv_db[self.STRATIFIED_SAMPLE_KEY]
            
            # Try to parse as JSON first
            if isinstance(stored_value, str):
                try:
                    df_dict = json.loads(stored_value)
                    logger.info(f"Retrieved stratified sample as JSON")
                except json.JSONDecodeError:
                    # Not valid JSON, might be pickle
                    logger.info("Stored value is not valid JSON, trying pickle")
                    try:
                        df_dict = pickle.loads(stored_value)
                        logger.info(f"Retrieved stratified sample from pickle")
                    except:
                        logger.error("Failed to deserialize stored value as pickle")
                        return None
            elif isinstance(stored_value, dict):
                # Direct dictionary (legacy format)
                df_dict = stored_value
                logger.info(f"Retrieved stratified sample as direct dictionary")
            else:
                # Try pickle for any other format
                try:
                    df_dict = pickle.loads(stored_value)
                    logger.info(f"Retrieved stratified sample from pickle")
                except:
                    logger.error("Unknown format for stratified sample")
                    return None
            
            # Reconstruct DataFrame from dictionary
            if isinstance(df_dict, dict) and 'data' in df_dict and 'columns' in df_dict:
                # Create DataFrame from records
                df = pd.DataFrame(df_dict['data'])
                
                # Reorder columns if needed and they all exist
                if set(df.columns).issuperset(set(df_dict['columns'])):
                    # Only keep columns that were in the original data
                    df = df[df_dict['columns']]
                
                logger.info(f"Retrieved stratified sample with {len(df)} rows from key-value store")
                return df
            elif isinstance(df_dict, pd.DataFrame):
                # Handle case where we stored a DataFrame directly
                logger.info(f"Retrieved stored DataFrame with {len(df_dict)} rows")
                return df_dict
            else:
                logger.warning("Invalid format for stratified sample in key-value store")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving stratified sample: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def store_embeddings(self, embeddings: np.ndarray, thread_id_map: Dict[str, int]):
        """
        Store embeddings array and thread ID mapping.
        
        Args:
            embeddings: NumPy array of embeddings
            thread_id_map: Dictionary mapping array indices to thread IDs
        """
        try:
            # Ensure thread_id_map is serializable (convert keys/values to strings if needed)
            clean_thread_map = {}
            for key, value in thread_id_map.items():
                # Ensure keys and values are JSON-serializable
                clean_thread_map[str(key)] = value
            
            # Convert NumPy array to a list format for storage
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
            # Check size
            estimated_size_mb = (len(str(embeddings_list)) + len(str(clean_thread_map))) / (1024 * 1024)
            logger.info(f"Estimated embeddings size: {estimated_size_mb:.2f} MB")
            
            if estimated_size_mb > 4.5:  # Leave some margin below the 5MB limit
                logger.warning(f"Embeddings size ({estimated_size_mb:.2f} MB) is approaching the key-value store limit")
                
                # If too large, split embeddings into chunks
                chunk_size = 100  # Number of embeddings per chunk
                num_embeddings = len(embeddings_list)
                num_chunks = (num_embeddings + chunk_size - 1) // chunk_size  # Ceiling division
                
                # Store chunk information in metadata
                self.db[f"{self.EMBEDDINGS_KEY}_meta"] = {
                    "chunks": num_chunks,
                    "shape": embeddings.shape if isinstance(embeddings, np.ndarray) else [len(embeddings_list), len(embeddings_list[0]) if embeddings_list else 0],
                    "timestamp": datetime.now().isoformat(),
                    "chunk_size": chunk_size,
                    "format": "chunked_list"
                }
                
                # Store each chunk
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, num_embeddings)
                    chunk = embeddings_list[start_idx:end_idx]
                    
                    # Convert chunk to JSON
                    chunk_json = json.dumps(chunk)
                    self.db[f"{self.EMBEDDINGS_KEY}_chunk_{i}"] = chunk_json
                
                logger.info(f"Stored embeddings in {num_chunks} chunks")
            else:
                # Store as a single value
                # Try JSON first
                try:
                    json_data = json.dumps(embeddings_list)
                    self.db[self.EMBEDDINGS_KEY] = json_data
                    
                    # Store metadata
                    self.db[f"{self.EMBEDDINGS_KEY}_meta"] = {
                        "chunks": 1,
                        "shape": embeddings.shape if isinstance(embeddings, np.ndarray) else [len(embeddings_list), len(embeddings_list[0]) if embeddings_list else 0],
                        "timestamp": datetime.now().isoformat(),
                        "format": "json"
                    }
                    
                    logger.info(f"Stored embeddings as JSON")
                except Exception as json_error:
                    logger.warning(f"Failed to store embeddings as JSON: {json_error}")
                    
                    # Fall back to pickle
                    serialized = pickle.dumps(embeddings)
                    self.db[self.EMBEDDINGS_KEY] = serialized
                    
                    # Store metadata
                    self.db[f"{self.EMBEDDINGS_KEY}_meta"] = {
                        "chunks": 1,
                        "shape": embeddings.shape if isinstance(embeddings, np.ndarray) else None,
                        "timestamp": datetime.now().isoformat(),
                        "format": "pickle"
                    }
                    
                    logger.info(f"Stored embeddings as pickle")
            
            # Store thread ID mapping (always as JSON)
            try:
                thread_map_json = json.dumps(clean_thread_map)
                self.db[self.THREAD_MAP_KEY] = thread_map_json
                logger.info(f"Stored thread ID map with {len(clean_thread_map)} entries as JSON")
            except Exception as e:
                logger.warning(f"Failed to store thread map as JSON: {e}")
                # Fall back to pickle
                self.db[self.THREAD_MAP_KEY] = pickle.dumps(clean_thread_map)
                logger.info(f"Stored thread ID map as pickle")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Retrieve embeddings and thread ID mapping.
        
        Returns:
            Tuple of (embeddings array, thread ID map dict) or (None, None) if not found
        """
        try:
            # Check if embeddings exist
            if f"{self.EMBEDDINGS_KEY}_meta" not in self.db:
                logger.info("No embeddings found in key-value store")
                return None, None
                
            # Get metadata
            meta = self.db[f"{self.EMBEDDINGS_KEY}_meta"]
            storage_format = meta.get("format", "pickle")  # Default to pickle for backward compatibility
            
            # Handle different storage formats
            if storage_format == "chunked_list":
                # Reassemble chunks
                chunks = []
                num_chunks = meta["chunks"]
                
                for i in range(num_chunks):
                    chunk_key = f"{self.EMBEDDINGS_KEY}_chunk_{i}"
                    if chunk_key in self.db:
                        # Parse JSON chunk
                        chunk_data = json.loads(self.db[chunk_key])
                        chunks.extend(chunk_data)
                    else:
                        logger.error(f"Missing chunk {i} for embeddings")
                        return None, None
                
                # Convert list back to numpy array
                embeddings = np.array(chunks)
                
            elif storage_format == "json":
                # Get single JSON value
                if self.EMBEDDINGS_KEY not in self.db:
                    logger.error("Embeddings metadata exists but value is missing")
                    return None, None
                    
                # Parse JSON embeddings
                embeddings_list = json.loads(self.db[self.EMBEDDINGS_KEY])
                embeddings = np.array(embeddings_list)
                
            else:
                # Handle legacy pickle format
                if meta.get("chunks", 1) > 1:
                    # Reassemble chunks
                    chunks = []
                    for i in range(meta["chunks"]):
                        chunk_key = f"{self.EMBEDDINGS_KEY}_{i}"
                        if chunk_key in self.db:
                            chunks.append(self.db[chunk_key])
                        else:
                            logger.error(f"Missing chunk {i} for embeddings")
                            return None, None
                    
                    serialized = b''.join(chunks)
                else:
                    # Get single value
                    if self.EMBEDDINGS_KEY not in self.db:
                        logger.error("Embeddings metadata exists but value is missing")
                        return None, None
                        
                    serialized = self.db[self.EMBEDDINGS_KEY]
                
                # Deserialize embeddings
                embeddings = pickle.loads(serialized)
            
            # Get thread ID map
            if self.THREAD_MAP_KEY not in self.db:
                logger.error("Thread ID map is missing")
                return embeddings, None
                
            try:
                # First try JSON format
                thread_id_map = json.loads(self.db[self.THREAD_MAP_KEY])
            except:
                # Fall back to pickle
                thread_id_map = pickle.loads(self.db[self.THREAD_MAP_KEY])
            
            # Convert string keys back to integers if needed
            try:
                numeric_thread_map = {}
                for key, value in thread_id_map.items():
                    if key.isdigit():
                        numeric_thread_map[int(key)] = value
                    else:
                        numeric_thread_map[key] = value
                thread_id_map = numeric_thread_map
            except:
                # Keep the original map if conversion fails
                pass
            
            logger.info(f"Retrieved embeddings with shape {embeddings.shape} and {len(thread_id_map)} thread mappings")
            return embeddings, thread_id_map
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return None, None
    
    def update_processing_state(self, state: Dict[str, Any]):
        """
        Update processing state information.
        
        Args:
            state: Dictionary containing state information
        """
        try:
            state["updated_at"] = datetime.now().isoformat()
            self.db[self.STATE_KEY] = json.dumps(state)
            logger.debug(f"Updated processing state: {state.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Error updating processing state: {e}")
            raise
    
    def get_processing_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current processing state.
        
        Returns:
            State dictionary or None if not set
        """
        try:
            if self.STATE_KEY not in self.db:
                return None
            return json.loads(self.db[self.STATE_KEY])
        except Exception as e:
            logger.error(f"Error getting processing state: {e}")
            raise