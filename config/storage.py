"""
Storage interfaces and implementations for Chanscope.

This module provides abstract interfaces and concrete implementations for
storage operations used in Chanscope, supporting both file-based storage
(Docker) and database storage (Replit).
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Awaitable
from pathlib import Path
from datetime import datetime, timedelta
from filelock import FileLock
import asyncio
import traceback

from config.env_loader import detect_environment
from config.settings import Config

# Configure logging
logger = logging.getLogger(__name__)

class CompleteDataStorage(ABC):
    """Abstract interface for complete dataset storage."""
    
    @abstractmethod
    async def store_data(self, df: pd.DataFrame) -> bool:
        """Store complete dataset."""
        pass
    
    @abstractmethod
    async def get_data(self, filter_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve complete dataset with optional filtering."""
        pass
    
    @abstractmethod
    async def is_data_fresh(self) -> bool:
        """Check if data is fresh based on timestamps."""
        pass
    
    @abstractmethod
    async def get_row_count(self) -> int:
        """Get the count of rows in the storage."""
        pass

class StratifiedSampleStorage(ABC):
    """Abstract interface for stratified sample storage."""
    
    @abstractmethod
    async def store_sample(self, df: pd.DataFrame) -> bool:
        """Store stratified sample."""
        pass
    
    @abstractmethod
    async def get_sample(self) -> Optional[pd.DataFrame]:
        """Retrieve stratified sample."""
        pass
    
    @abstractmethod
    async def sample_exists(self) -> bool:
        """Check if stratified sample exists."""
        pass

class EmbeddingStorage(ABC):
    """Abstract interface for embedding storage."""
    
    @abstractmethod
    async def store_embeddings(self, embeddings: np.ndarray, thread_id_map: Dict[str, int]) -> bool:
        """Store embeddings and thread ID map."""
        pass
    
    @abstractmethod
    async def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Retrieve embeddings and thread ID map."""
        pass
    
    @abstractmethod
    async def embeddings_exist(self) -> bool:
        """Check if embeddings exist."""
        pass

class StateManager(ABC):
    """Abstract interface for state management."""
    
    @abstractmethod
    async def update_state(self, state: Dict[str, Any]) -> None:
        """Update processing state."""
        pass
    
    @abstractmethod
    async def get_state(self) -> Optional[Dict[str, Any]]:
        """Get current processing state."""
        pass
    
    @abstractmethod
    async def mark_operation_start(self, operation: str) -> None:
        """Mark the start of an operation."""
        pass
    
    @abstractmethod
    async def mark_operation_complete(self, operation: str, result: Any = None) -> None:
        """Mark the completion of an operation."""
        pass
    
    @abstractmethod
    async def is_operation_in_progress(self, operation: str) -> bool:
        """Check if an operation is in progress."""
        pass

# File-based implementations
class FileCompleteDataStorage(CompleteDataStorage):
    """File-based implementation of complete dataset storage."""
    
    def __init__(self, config):
        self.config = config
        self.complete_data_path = Path(config.root_data_path) / "complete_data.csv"
    
    async def store_data(self, df: pd.DataFrame) -> bool:
        """Store complete dataset to CSV file."""
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return False
        
        try:
            # Ensure directory exists
            self.complete_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(self.complete_data_path, index=False)
            logger.info(f"Stored {len(df)} rows to {self.complete_data_path}")
            return True
        except Exception as e:
            logger.error(f"Error storing complete data: {e}")
            return False
    
    async def get_data(self, filter_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve complete dataset from CSV file with optional filtering."""
        if not self.complete_data_path.exists():
            logger.warning(f"Complete data file not found at {self.complete_data_path}")
            return pd.DataFrame()
        
        try:
            # Read CSV
            df = pd.read_csv(self.complete_data_path)
            
            # Apply filtering if needed
            if filter_date and 'posted_date_time' in df.columns:
                try:
                    filter_datetime = pd.to_datetime(filter_date, utc=True)
                    df['posted_date_time'] = pd.to_datetime(df['posted_date_time'], utc=True)
                    df = df[df['posted_date_time'] >= filter_datetime]
                    logger.info(f"Filtered data to {len(df)} rows after {filter_date}")
                except Exception as e:
                    logger.warning(f"Error applying date filter: {e}")
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving complete data: {e}")
            return pd.DataFrame()
    
    async def is_data_fresh(self) -> bool:
        """Check if data is fresh based on file modification time."""
        if not self.complete_data_path.exists():
            return False
        
        try:
            # Get file modification time
            mod_time = datetime.fromtimestamp(self.complete_data_path.stat().st_mtime)
            
            # Consider fresh if modified within the last day
            return (datetime.now() - mod_time) < timedelta(days=1)
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return False
    
    async def get_row_count(self) -> int:
        """Get row count from CSV file."""
        if not self.complete_data_path.exists():
            return 0
        
        try:
            # Get row count efficiently without loading entire file
            with open(self.complete_data_path, 'r') as f:
                # Subtract 1 for header row
                return sum(1 for _ in f) - 1
        except Exception as e:
            logger.error(f"Error getting row count: {e}")
            return 0

class FileStratifiedSampleStorage(StratifiedSampleStorage):
    """File-based implementation of stratified sample storage."""
    
    def __init__(self, config):
        self.config = config
        self.stratified_path = Path(config.stratified_data_path) / "stratified_sample.csv"
    
    async def store_sample(self, df: pd.DataFrame) -> bool:
        """Store stratified sample to CSV file."""
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return False
        
        try:
            # Ensure directory exists
            self.stratified_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(self.stratified_path, index=False)
            logger.info(f"Stored stratified sample with {len(df)} rows to {self.stratified_path}")
            return True
        except Exception as e:
            logger.error(f"Error storing stratified sample: {e}")
            return False
    
    async def get_sample(self) -> Optional[pd.DataFrame]:
        """Retrieve stratified sample from CSV file."""
        if not self.stratified_path.exists():
            logger.warning(f"Stratified sample not found at {self.stratified_path}")
            return None
        
        try:
            # Read CSV
            df = pd.read_csv(self.stratified_path)
            logger.info(f"Retrieved stratified sample with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error retrieving stratified sample: {e}")
            return None
    
    async def sample_exists(self) -> bool:
        """Check if stratified sample file exists."""
        return self.stratified_path.exists()

class FileEmbeddingStorage(EmbeddingStorage):
    """File-based implementation of embedding storage."""
    
    def __init__(self, config):
        self.config = config
        self.embeddings_path = Path(config.stratified_data_path) / "embeddings.npz"
        self.thread_id_map_path = Path(config.stratified_data_path) / "thread_id_map.json"
    
    async def store_embeddings(self, embeddings: np.ndarray, thread_id_map: Dict[str, int]) -> bool:
        """Store embeddings to NPZ file and thread ID map to JSON file."""
        if embeddings.size == 0 or not thread_id_map:
            logger.warning("Empty embeddings or thread ID map, nothing to store")
            return False
        
        try:
            # Ensure directory exists
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save embeddings
            np.savez_compressed(self.embeddings_path, embeddings=embeddings)
            
            # Save thread ID map
            with open(self.thread_id_map_path, 'w') as f:
                json.dump(thread_id_map, f)
            
            logger.info(f"Stored embeddings with shape {embeddings.shape} and {len(thread_id_map)} thread IDs")
            return True
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    async def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Retrieve embeddings from NPZ file and thread ID map from JSON file."""
        if not self.thread_id_map_path.exists():
            logger.warning(f"Missing thread ID map file: {self.thread_id_map_path}")
            return None, None
        
        # Check for embeddings file
        embeddings_file_exists = self.embeddings_path.exists()
        embeddings_npy_file = self.embeddings_path.with_suffix('.npy')
        embeddings_npy_exists = embeddings_npy_file.exists()
        
        if not embeddings_file_exists and not embeddings_npy_exists:
            logger.warning(f"Missing embedding files: {self.embeddings_path} or {embeddings_npy_file}")
            return None, None
        
        try:
            # Load embeddings - try both .npz and .npy formats
            embeddings = None
            
            # First try .npz format (preferred)
            if embeddings_file_exists:
                try:
                    with np.load(self.embeddings_path) as data:
                        embeddings = data['embeddings']
                    logger.info(f"Successfully loaded embeddings from .npz file")
                except Exception as e:
                    logger.warning(f"Error loading .npz embeddings: {e}. Will try .npy format.")
            
            # If .npz failed or doesn't exist, try .npy format
            if embeddings is None and embeddings_npy_exists:
                try:
                    embeddings = np.load(embeddings_npy_file)
                    logger.info(f"Successfully loaded embeddings from .npy file")
                except Exception as e:
                    logger.error(f"Error loading .npy embeddings: {e}")
                    return None, None
            
            # If we still don't have embeddings, return None
            if embeddings is None:
                logger.error("Failed to load embeddings from either .npz or .npy format")
                return None, None
            
            # Load thread ID map
            with open(self.thread_id_map_path, 'r') as f:
                thread_id_map = json.load(f)
            
            logger.info(f"Retrieved embeddings with shape {embeddings.shape} and {len(thread_id_map)} thread IDs")
            return embeddings, thread_id_map
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return None, None
    
    async def embeddings_exist(self) -> bool:
        """Check if embeddings and thread ID map files exist."""
        embeddings_npz_exists = self.embeddings_path.exists()
        embeddings_npy_exists = self.embeddings_path.with_suffix('.npy').exists()
        thread_map_exists = self.thread_id_map_path.exists()
        
        # Return True if either embeddings format exists along with the thread map
        return (embeddings_npz_exists or embeddings_npy_exists) and thread_map_exists

class FileStateManager(StateManager):
    """File-based implementation of state management."""
    
    def __init__(self, config):
        self.config = config
        self.state_file = Path(config.root_data_path) / ".initialization_state"
        self.in_progress_marker = Path(config.root_data_path) / ".initialization_in_progress"
        self.completion_marker = Path(config.root_data_path) / ".initialization_complete"
        
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """Update state in JSON file."""
        try:
            # Add timestamp
            state["updated_at"] = datetime.now().isoformat()
            
            # Write state
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            
            logger.debug(f"Updated state: {state.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Error updating state: {e}")
    
    async def get_state(self) -> Optional[Dict[str, Any]]:
        """Get state from JSON file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving state: {e}")
            return None
    
    async def mark_operation_start(self, operation: str) -> None:
        """Mark the start of an operation by creating a marker file."""
        try:
            # Create in-progress marker
            self.in_progress_marker.touch()
            
            # Update state
            await self.update_state({
                "status": "in_progress",
                "operation": operation,
                "started_at": datetime.now().isoformat()
            })
            
            logger.info(f"Marked operation start: {operation}")
        except Exception as e:
            logger.error(f"Error marking operation start: {e}")
    
    async def mark_operation_complete(self, operation: str, result: Any = None) -> None:
        """Mark the completion of an operation by updating state and markers."""
        try:
            # Update state
            await self.update_state({
                "status": "complete",
                "operation": operation,
                "completed_at": datetime.now().isoformat(),
                "result": result
            })
            
            # Create completion marker
            self.completion_marker.touch()
            
            # Remove in-progress marker if it exists
            if self.in_progress_marker.exists():
                self.in_progress_marker.unlink()
            
            logger.info(f"Marked operation complete: {operation}")
        except Exception as e:
            logger.error(f"Error marking operation complete: {e}")
    
    async def is_operation_in_progress(self, operation: str) -> bool:
        """Check if an operation is in progress based on marker file."""
        if not self.in_progress_marker.exists():
            return False
        
        try:
            # Check if in-progress marker is stale (older than 30 minutes)
            mod_time = datetime.fromtimestamp(self.in_progress_marker.stat().st_mtime)
            if (datetime.now() - mod_time) > timedelta(minutes=30):
                logger.warning(f"Stale in-progress marker found, removing")
                self.in_progress_marker.unlink()
                return False
            
            # Check state file for operation
            state = await self.get_state()
            if state and state.get('operation') == operation and state.get('status') == 'in_progress':
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking operation status: {e}")
            return False

# Replit implementations (use our existing classes from replit.py)
if os.environ.get('REPLIT_ENV', 'false').lower() in ('true', 'replit', '1', 'yes'):
    from config.replit import PostgresDB, KeyValueStore, DatabaseConnectionError

class ReplitCompleteDataStorage(CompleteDataStorage):
    """Replit PostgreSQL implementation of complete dataset storage."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        from config.replit import PostgresDB
        self.config = config
        self.db = PostgresDB()
        
        # Get chunk settings from config
        chunk_settings = Config.get_chunk_settings()
        self.processing_chunk_size = chunk_settings.get('processing_chunk_size', 25000)
        
        # Get column settings from config
        column_settings = Config.get_column_settings()
        self.time_column = column_settings.get('time_column', 'posted_date_time')
        
        # Get board ID from config if available
        self.board_id = getattr(config, 'board_id', None)
        
        logger.info(f"Initialized ReplitCompleteDataStorage with chunk_size={self.processing_chunk_size}, "
                   f"time_column={self.time_column}, board_id={self.board_id}")
    
    async def prepare_data(self) -> bool:
        """Prepare the complete dataset in PostgreSQL database."""
        try:
            # Initialize schema if needed
            self.db.initialize_schema()
            logger.info("Database schema initialized")
            
            # Calculate retention period from current date
            current_time = pd.Timestamp.now(tz='UTC')
            retention_settings = Config.get_retention_settings()
            retention_days = retention_settings.get('retention_days', 30)  # Default to 30 days if not specified
            
            # Calculate start_time as current_time minus retention_days
            start_time = current_time - pd.Timedelta(days=retention_days)
            
            logger.info(f"Using retention period of {retention_days} days")
            logger.info(f"Data range: from {start_time.isoformat()} to {current_time.isoformat()}")
            
            # Import S3 handler here to avoid issues if S3 modules aren't available
            from knowledge_agents.data_processing.cloud_handler import S3Handler, load_all_csv_data_from_s3
            
            # Check S3 connectivity
            s3_handler = S3Handler()
            if not s3_handler.is_configured:
                logger.error("S3 is not properly configured")
                return False
            
            # Initialize counters
            record_count = 0
            files_processed = 0
            files_skipped = 0
            
            # Use the _get_filtered_csv_files method directly to get list of relevant files
            csv_files = s3_handler._get_filtered_csv_files(latest_date=start_time)
            if not csv_files:
                logger.error("No CSV files found in S3 for the specified date range")
                return False
            
            logger.info(f"Found {len(csv_files)} CSV files in S3 within the retention period")
            
            # Process each file that might contain data in our date range
            for file_path in csv_files:
                logger.info(f"Processing file: {file_path}")
                try:
                    # Create generator for this specific file
                    file_data_generator = load_all_csv_data_from_s3(
                        latest_date_processed=start_time.isoformat(),
                        chunk_size=self.processing_chunk_size,
                        board_id=self.board_id
                    )
                    
                    file_record_count = 0
                    async for chunk in file_data_generator:
                        # Process date column
                        chunk[self.time_column] = pd.to_datetime(
                            chunk[self.time_column], 
                            format='mixed',
                            utc=True,
                            errors='coerce'
                        )
                        
                        # Filter by date range (keep data between start_time and current_time)
                        date_mask = (chunk[self.time_column] >= start_time) & (chunk[self.time_column] <= current_time)
                        filtered_chunk = chunk[date_mask]
                        
                        if not filtered_chunk.empty:
                            file_record_count += len(filtered_chunk)
                            # Store chunk in database
                            await self.store_data(filtered_chunk)
                            logger.info(f"Processed chunk with {len(filtered_chunk)} rows from file {file_path}")
                        
                        # Yield to other tasks
                        await asyncio.sleep(0)
                    
                    # Update counters
                    record_count += file_record_count
                    files_processed += 1
                    logger.info(f"Completed file {file_path} with {file_record_count} records in retention period")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    logger.error(traceback.format_exc())
                    # Continue with other files
            
            # Log final stats
            logger.info(f"Data loading summary:")
            logger.info(f"- Files processed: {files_processed}")
            logger.info(f"- Files skipped: {files_skipped}")
            logger.info(f"- Total records loaded: {record_count}")
            
            if record_count == 0:
                logger.error("No data fetched from S3 within retention period")
                return False
            
            logger.info(f"Successfully processed and stored {record_count} total rows")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def store_data(self, df: pd.DataFrame) -> bool:
        """Store complete dataset in PostgreSQL database."""
        try:
            # Use the new sync method to only add new data
            rows_added = self.db.sync_data_from_dataframe(df)
            logger.info(f"Synchronized complete data with PostgreSQL, added {rows_added} new rows")
            return True
        except Exception as e:
            logger.error(f"Error storing complete data: {e}")
            return False
    
    async def get_data(self, filter_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve complete dataset from PostgreSQL database."""
        try:
            # Use the core database method
            df = self.db.get_complete_data(filter_date)
            
            # Ensure the time column is datetime type
            if self.time_column in df.columns and not pd.api.types.is_datetime64_dtype(df[self.time_column]):
                logger.info(f"Converting {self.time_column} to datetime type in storage layer")
                df[self.time_column] = pd.to_datetime(df[self.time_column], errors='coerce')
                # Drop rows with invalid dates
                invalid_dates = df[self.time_column].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"Dropping {invalid_dates} rows with invalid {self.time_column} values")
                    df = df.dropna(subset=[self.time_column])
            
            logger.info(f"Retrieved {len(df)} rows from PostgreSQL database")
            return df
        except Exception as e:
            logger.error(f"Error retrieving complete data: {e}")
            return pd.DataFrame()
    
    async def is_data_fresh(self) -> bool:
        """Check if data is fresh based on database timestamps."""
        try:
            # Use the new check method
            needs_update, last_updated = self.db.check_data_needs_update()
            if last_updated:
                logger.info(f"PostgreSQL data last updated at {last_updated}")
            return not needs_update
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return False
    
    async def get_row_count(self) -> int:
        """Get row count from PostgreSQL database."""
        try:
            # Use the core database method
            count = self.db.get_row_count()
            logger.info(f"PostgreSQL database has {count} rows")
            return count
        except Exception as e:
            logger.error(f"Error getting row count: {e}")
            return 0

class ReplitStratifiedSampleStorage(StratifiedSampleStorage):
    """Object Storage-based implementation of stratified sample storage for Replit."""
    
    def __init__(self, config):
        self.config = config
        self.default_bucket = "knowledge-agent-data"  # Default bucket name
        self.stratified_key = "stratified_sample.json"
        self.metadata_key = "stratified_metadata.json"
    
    async def store_sample(self, df: pd.DataFrame) -> bool:
        """Store stratified sample to Object Storage."""
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return False
        
        try:
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                # Try to initialize with default bucket
                object_client = Client()
            except ValueError as bucket_error:
                # If error mentions 'no default bucket', use our default
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}'. "
                                  f"Please configure a bucket in .replit file with: bucket = \"{self.default_bucket}\"")
                    try:
                        # Try to use the default bucket name
                        object_client = Client(bucket=self.default_bucket)
                    except Exception as e:
                        logger.error(f"Failed to initialize Object Storage with default bucket: {e}")
                        return False
                else:
                    # Re-raise other errors
                    raise
            
            # Create a copy of the DataFrame for serialization
            df_copy = df.copy()
            
            # Convert all datetime/timestamp columns to ISO format strings
            datetime_columns = df_copy.select_dtypes(
                include=['datetime64[ns]', 'datetime64[ns, UTC]']
            ).columns
            
            for col in datetime_columns:
                df_copy[col] = df_copy[col].apply(
                    lambda x: x.isoformat() if pd.notnull(x) else None
                )
            
            # Convert any remaining non-serializable types to strings
            for col in df_copy.columns:
                if df_copy[col].dtype.name not in ['object', 'str', 'int64', 'float64', 'bool']:
                    df_copy[col] = df_copy[col].astype(str)
            
            # Convert DataFrame to records for JSON serialization
            records = df_copy.to_dict('records')
            
            # Create metadata with column types for proper reconstruction
            metadata = {
                'columns': list(df_copy.columns),
                'datetime_columns': list(datetime_columns),
                'row_count': len(df_copy),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store data and metadata in Object Storage
            object_client.upload_from_text(self.stratified_key, json.dumps(records))
            object_client.upload_from_text(self.metadata_key, json.dumps(metadata))
            
            logger.info(f"Successfully stored stratified sample with {len(df)} rows to Object Storage")
            return True
            
        except Exception as e:
            logger.error(f"Error storing stratified sample in Object Storage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def get_sample(self) -> Optional[pd.DataFrame]:
        """Retrieve stratified sample from Object Storage."""
        try:
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                # Try to initialize with default bucket
                object_client = Client()
            except ValueError as bucket_error:
                # If error mentions 'no default bucket', use our default
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}'")
                    try:
                        object_client = Client(bucket=self.default_bucket)
                    except Exception as e:
                        logger.error(f"Failed to initialize Object Storage with default bucket: {e}")
                        return None
                else:
                    raise
            
            # Check if files exist in Object Storage
            objects = object_client.list()
            object_names = [obj.name for obj in objects]
            
            if self.stratified_key not in object_names or self.metadata_key not in object_names:
                logger.warning("Stratified sample or metadata not found in Object Storage")
                return None
            
            # Load data and metadata
            try:
                records = json.loads(object_client.download_as_text(self.stratified_key))
                metadata = json.loads(object_client.download_as_text(self.metadata_key))
            except json.JSONDecodeError:
                logger.error("Error decoding stored data")
                return None
            
            # Create DataFrame from records
            df = pd.DataFrame(records)
            
            # Reorder columns if needed
            if 'columns' in metadata and set(df.columns).issuperset(set(metadata['columns'])):
                df = df[metadata['columns']]
            
            # Convert datetime columns back to datetime type
            datetime_columns = metadata.get('datetime_columns', [])
            for col in datetime_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], utc=True)
                    except Exception as e:
                        logger.warning(f"Error converting {col} to datetime: {e}")
            
            logger.info(f"Retrieved stratified sample with {len(df)} rows from Object Storage")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving stratified sample from Object Storage: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def sample_exists(self) -> bool:
        """Check if stratified sample exists in Object Storage."""
        try:
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                # Try to initialize with default bucket
                object_client = Client()
            except ValueError as bucket_error:
                # If error mentions 'no default bucket', use our default
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}'")
                    try:
                        object_client = Client(bucket=self.default_bucket)
                    except Exception as e:
                        logger.error(f"Failed to initialize Object Storage with default bucket: {e}")
                        return False
                else:
                    raise
            
            # List objects and check if stratified sample and metadata exist
            objects = object_client.list()
            object_names = [obj.name for obj in objects]
            
            exists = self.stratified_key in object_names and self.metadata_key in object_names
            if exists:
                try:
                    metadata = json.loads(object_client.download_as_text(self.metadata_key))
                    logger.info(f"Found stratified sample with {metadata.get('row_count', 'unknown')} rows from {metadata.get('timestamp', 'unknown')}")
                except:
                    logger.warning("Found stratified sample but couldn't read metadata")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking if sample exists in Object Storage: {e}")
            logger.error(traceback.format_exc())
            return False

class ReplitObjectEmbeddingStorage(EmbeddingStorage):
    """Object Storage-based implementation of embedding storage for Replit."""
    
    def __init__(self, config):
        """Initialize Object Storage client."""
        self.config = config
        self.embeddings_key = "embeddings.npy"
        self.thread_map_key = "thread_id_map.json"
        self.replit_kv = None  # For state tracking only
        self.default_bucket = "knowledge-agent-embeddings"  # Default bucket name
    
    async def store_embeddings(self, embeddings: np.ndarray, thread_id_map: Dict[str, int]) -> bool:
        """Store embeddings to Object Storage and thread ID map."""
        if embeddings.size == 0 or not thread_id_map:
            logger.warning("Empty embeddings or thread ID map, nothing to store")
            return False
        
        try:
            # Validate embedding format
            if len(embeddings.shape) != 2:
                logger.error(f"Invalid embedding shape: {embeddings.shape}. Expected 2-dimensional array.")
                return False
                
            # Validate embedding values
            if not np.isfinite(embeddings).all():
                invalid_indices = np.where(~np.isfinite(embeddings))[0]
                logger.error(f"Invalid values (inf or nan) found in embeddings at indices: {invalid_indices}")
                return False
                
            # Validate embedding dimensions are consistent
            expected_dim = embeddings.shape[1]
            for idx, row in enumerate(embeddings):
                if len(row) != expected_dim:
                    logger.warning(f"Unexpected embedding format at index {idx}: Expected dimension {expected_dim}, got {len(row)}")
                    return False
            
            logger.info(f"Validated embeddings with shape {embeddings.shape} for storage")
            
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                # Try to initialize with default bucket
                logger.info("Initializing Replit Object Storage client")
                object_client = Client()
                logger.info(f"Successfully initialized Object Storage client with default bucket")
            except ValueError as bucket_error:
                # If error mentions 'no default bucket', use our default
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}'. "
                                  f"Please configure a bucket in .replit file with: bucket = \"{self.default_bucket}\"")
                    try:
                        # Try to use the default bucket name
                        object_client = Client(bucket=self.default_bucket)
                        logger.info(f"Successfully initialized Object Storage client with bucket: {self.default_bucket}")
                    except Exception as e:
                        logger.error(f"Failed to initialize Object Storage with default bucket: {e}")
                        return False
                else:
                    # Re-raise other errors
                    raise
            
            # Ensure consistent thread ID format (convert all to strings)
            clean_thread_map = {str(k).strip(): v for k, v in thread_id_map.items()}
            
            # Add metadata to thread map
            thread_map_with_meta = {
                'thread_ids': clean_thread_map,
                'embedding_shape': embeddings.shape,
                'timestamp': datetime.now().isoformat(),
                'format_version': '1.0'
            }
            
            # Save thread map to Object Storage
            logger.info(f"Uploading thread ID map with {len(clean_thread_map)} entries to Object Storage at key: {self.thread_map_key}")
            try:
                object_client.upload_from_text(self.thread_map_key, json.dumps(thread_map_with_meta))
                logger.info(f"Successfully uploaded thread ID map to Object Storage")
            except Exception as e:
                logger.error(f"Failed to upload thread ID map to Object Storage: {e}")
                return False
            
            # Save embeddings to Object Storage
            # First, save embeddings to a temporary file
            temp_file = "/tmp/embeddings.npy"
            try:
                logger.info(f"Saving embeddings with shape {embeddings.shape} to temporary file: {temp_file}")
                np.save(temp_file, embeddings)
                logger.info(f"Successfully saved embeddings to temporary file")
            except Exception as e:
                logger.error(f"Failed to save embeddings to temporary file: {e}")
                return False
            
            # Upload the file to Object Storage using upload_from_bytes
            try:
                logger.info(f"Reading embedding bytes from temporary file")
                with open(temp_file, "rb") as f:
                    embedding_bytes = f.read()
                    bytes_size = len(embedding_bytes) / (1024 * 1024)  # Size in MB
                    logger.info(f"Uploading {bytes_size:.2f} MB of embedding data to Object Storage at key: {self.embeddings_key}")
                    object_client.upload_from_bytes(self.embeddings_key, embedding_bytes)
                    logger.info(f"Successfully uploaded {bytes_size:.2f} MB of embedding data to Object Storage")
            except Exception as e:
                logger.error(f"Failed to upload embeddings to Object Storage: {e}")
                return False
            
            # Clean up temp file
            try:
                os.remove(temp_file)
                logger.info(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            # Verify upload was successful by listing objects
            try:
                objects = object_client.list()
                object_names = [obj.name for obj in objects]
                
                if self.embeddings_key in object_names and self.thread_map_key in object_names:
                    logger.info(f"Verified both embeddings and thread map are present in Object Storage")
                else:
                    missing = []
                    if self.embeddings_key not in object_names:
                        missing.append(self.embeddings_key)
                    if self.thread_map_key not in object_names:
                        missing.append(self.thread_map_key)
                        
                    logger.warning(f"Verification failed, missing objects: {missing}")
                    return False
            except Exception as e:
                logger.warning(f"Could not verify object existence: {e}")
            
            logger.info(f"Successfully stored embeddings with shape {embeddings.shape} to Object Storage")
            return True
        except Exception as e:
            logger.error(f"Error storing embeddings in Object Storage: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Retrieve embeddings and thread ID map from Object Storage."""
        try:
            logger.info("Retrieving embeddings from Object Storage")
            
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                # Try to initialize with default bucket
                logger.info("Initializing Object Storage client for retrieval")
                object_client = Client()
                logger.info("Successfully initialized Object Storage client with default bucket")
            except ValueError as bucket_error:
                # If error mentions 'no default bucket', use our default
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}' for retrieval")
                    try:
                        # Try to use the default bucket name
                        object_client = Client(bucket=self.default_bucket)
                        logger.info(f"Successfully initialized Object Storage client with bucket: {self.default_bucket}")
                    except Exception as e:
                        logger.error(f"Failed to initialize Object Storage with default bucket for retrieval: {e}")
                        return None, None
                else:
                    # Re-raise other errors
                    raise
            
            # Check if files exist in Object Storage
            logger.info("Checking if embeddings and thread map exist in Object Storage")
            objects = object_client.list()
            object_names = [obj.name for obj in objects]
            
            if self.embeddings_key not in object_names:
                logger.warning(f"Embeddings not found in Object Storage: {self.embeddings_key}")
                return None, None
            
            if self.thread_map_key not in object_names:
                logger.warning(f"Thread map not found in Object Storage: {self.thread_map_key}")
                return None, None
            
            # Download thread ID map
            logger.info(f"Downloading thread ID map from key: {self.thread_map_key}")
            try:
                thread_map_json = object_client.download_as_text(self.thread_map_key)
                thread_map_data = json.loads(thread_map_json)
                logger.info(f"Successfully downloaded thread ID map")
            except Exception as e:
                logger.error(f"Failed to download or parse thread ID map: {e}")
                return None, None
            
            # Handle both old and new format
            if isinstance(thread_map_data, dict) and 'thread_ids' in thread_map_data:
                thread_id_map = thread_map_data['thread_ids']
                expected_shape = thread_map_data.get('embedding_shape')
                logger.info(f"Retrieved thread map with metadata format v1+ (expected shape: {expected_shape})")
            else:
                thread_id_map = thread_map_data
                expected_shape = None
                logger.info(f"Retrieved thread map with legacy format (no shape metadata)")
            
            # Ensure consistent thread ID format
            thread_id_map = {str(k).strip(): v for k, v in thread_id_map.items()}
            logger.info(f"Thread ID map contains {len(thread_id_map)} entries")
            
            # Download embeddings using download_as_bytes
            temp_file = "/tmp/embeddings.npy"
            try:
                logger.info(f"Downloading embedding data from key: {self.embeddings_key}")
                embedding_bytes = object_client.download_as_bytes(self.embeddings_key)
                bytes_size = len(embedding_bytes) / (1024 * 1024)  # Size in MB
                logger.info(f"Downloaded {bytes_size:.2f} MB of embedding data")
                
                # Write bytes to temporary file
                with open(temp_file, "wb") as f:
                    f.write(embedding_bytes)
                
                # Load embeddings from temp file
                logger.info(f"Loading embeddings from temp file: {temp_file}")
                embeddings = np.load(temp_file)
                logger.info(f"Successfully loaded embeddings with shape {embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to download or load embeddings: {e}")
                logger.error(traceback.format_exc())
                return None, None
            
            # Validate embeddings
            if expected_shape and embeddings.shape != tuple(expected_shape):
                logger.warning(f"Embedding shape mismatch. Expected {expected_shape}, got {embeddings.shape}")
            
            if not np.isfinite(embeddings).all():
                logger.warning("Invalid values in embeddings (inf or nan detected)")
            
            # Clean up temp file
            try:
                os.remove(temp_file)
                logger.info(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            logger.info(f"Retrieved embeddings with shape {embeddings.shape} from Object Storage")
            return embeddings, thread_id_map
        except Exception as e:
            logger.error(f"Error retrieving embeddings from Object Storage: {e}")
            logger.error(traceback.format_exc())
            return None, None
    
    async def embeddings_exist(self) -> bool:
        """Check if embeddings exist in Object Storage."""
        try:
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                # Try to initialize with default bucket
                object_client = Client()
            except ValueError as bucket_error:
                # If error mentions 'no default bucket', use our default
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}' for retrieval")
                    try:
                        # Try to use the default bucket name
                        object_client = Client(bucket=self.default_bucket)
                    except Exception as e:
                        logger.error(f"Failed to initialize Object Storage with default bucket for retrieval: {e}")
                        return False
                else:
                    # Re-raise other errors
                    raise
            
            # List objects and check if embeddings and thread map exist
            objects = object_client.list()
            object_names = [obj.name for obj in objects]
            
            exists = self.embeddings_key in object_names and self.thread_map_key in object_names
            if exists:
                logger.info("Embeddings found in Object Storage")
            else:
                logger.info("Embeddings not found in Object Storage")
            
            return exists
        except Exception as e:
            logger.error(f"Error checking if embeddings exist in Object Storage: {e}")
            return False

class ReplitStateManager(StateManager):
    """Replit Key-Value store implementation of state management."""
    
    def __init__(self, config):
        self.config = config
        self.kv_store = KeyValueStore()
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """Update state in Key-Value store."""
        try:
            self.kv_store.update_processing_state(state)
        except Exception as e:
            logger.error(f"Error updating state: {e}")
    
    async def get_state(self) -> Optional[Dict[str, Any]]:
        """Get state from Key-Value store."""
        try:
            return self.kv_store.get_processing_state()
        except Exception as e:
            logger.error(f"Error retrieving state: {e}")
            return None
    
    async def mark_operation_start(self, operation: str) -> None:
        """Mark the start of an operation in Key-Value store."""
        try:
            await self.update_state({
                "status": "in_progress",
                "operation": operation,
                "started_at": datetime.now().isoformat()
            })
            logger.info(f"Marked operation start: {operation}")
        except Exception as e:
            logger.error(f"Error marking operation start: {e}")
    
    async def mark_operation_complete(self, operation: str, result: Any = None) -> None:
        """Mark the completion of an operation in Key-Value store."""
        try:
            await self.update_state({
                "status": "complete",
                "operation": operation,
                "completed_at": datetime.now().isoformat(),
                "result": result
            })
            logger.info(f"Marked operation complete: {operation}")
        except Exception as e:
            logger.error(f"Error marking operation complete: {e}")
    
    async def is_operation_in_progress(self, operation: str) -> bool:
        """Check if an operation is in progress in Key-Value store."""
        try:
            state = await self.get_state()
            if state and state.get('operation') == operation and state.get('status') == 'in_progress':
                # Check if operation is stale (older than 30 minutes)
                started_at = state.get('started_at')
                if started_at:
                    start_time = datetime.fromisoformat(started_at)
                    if (datetime.now() - start_time) > timedelta(minutes=30):
                        logger.warning(f"Stale operation found: {operation}, marking as failed")
                        await self.update_state({
                            "status": "failed",
                            "operation": operation,
                            "error": "Operation timed out"
                        })
                        return False
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking operation status: {e}")
            return False

class StorageFactory:
    """Factory for creating storage implementations based on environment."""
    
    @staticmethod
    def create(config, env_type: str = None) -> Dict[str, Any]:
        """
        Create storage implementations based on environment type.
        
        Args:
            config: The configuration object
            env_type: Optional environment type override ('replit' or 'docker')
            
        Returns:
            Dictionary containing instantiated storage implementations:
            {
                'complete_data': CompleteDataStorage implementation,
                'stratified_sample': StratifiedSampleStorage implementation,
                'embeddings': EmbeddingStorage implementation,
                'state': StateManager implementation
            }
        """
        if env_type is None:
            # Use the centralized detection function
            env_type = detect_environment()
        
        logger.info(f"Creating storage implementations for environment: {env_type}")
        
        if env_type.lower() == 'replit':
            return {
                'complete_data': ReplitCompleteDataStorage(config),
                'stratified_sample': ReplitStratifiedSampleStorage(config),
                'embeddings': ReplitObjectEmbeddingStorage(config),
                'state': ReplitStateManager(config)
            }
        else:
            return {
                'complete_data': FileCompleteDataStorage(config),
                'stratified_sample': FileStratifiedSampleStorage(config),
                'embeddings': FileEmbeddingStorage(config),
                'state': FileStateManager(config)
            } 