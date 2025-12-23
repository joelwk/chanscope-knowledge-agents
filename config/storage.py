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

# In-memory Object Storage fallback for tests or missing replit deps
class _InMemoryObjectClient:
    _store: Dict[str, Union[str, bytes]] = {}

    def upload_from_text(self, key: str, text: str) -> None:
        self._store[key] = text

    def upload_from_bytes(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def download_as_text(self, key: str) -> str:
        value = self._store[key]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def download_as_bytes(self, key: str) -> bytes:
        value = self._store[key]
        if isinstance(value, str):
            return value.encode("utf-8")
        return value

    def list(self):
        from types import SimpleNamespace
        return [SimpleNamespace(name=key) for key in self._store.keys()]


class _InMemoryStateStore:
    """Minimal in-memory state store for tests."""

    def __init__(self):
        self._state: Optional[Dict[str, Any]] = None

    def update_processing_state(self, state: Dict[str, Any]) -> None:
        self._state = dict(state) if state is not None else None

    def get_processing_state(self) -> Optional[Dict[str, Any]]:
        return self._state

# Replit dependency placeholders (allow imports/mocking without hard failures)
PostgresDB = None
KeyValueStore = None
DatabaseConnectionError = Exception
db = None

try:
    from config.replit import (
        PostgresDB as _PostgresDB,
        KeyValueStore as _KeyValueStore,
        DatabaseConnectionError as _DatabaseConnectionError,
    )
    PostgresDB = _PostgresDB
    KeyValueStore = _KeyValueStore
    DatabaseConnectionError = _DatabaseConnectionError
    try:
        from replit import db as _db  # Optional; used for tests/mocking
        db = _db
    except Exception:
        db = None
except Exception as exc:
    logger.warning("Replit dependencies unavailable: %s", exc)

# Patch for Google Cloud Storage credentials to fix the universe_domain issue
def patch_gcs_credentials():
    """Apply a global patch to Google Cloud Storage credentials to add universe_domain attribute.
    
    This fixes the AttributeError: 'Credentials' object has no attribute 'universe_domain'
    that occurs when using newer google-cloud-storage versions with replit-object-storage.
    """
    try:
        import google.auth.credentials as credentials_module

        credentials_cls = credentials_module.Credentials

        # Skip patch if the attribute already exists or patch already applied
        if hasattr(credentials_cls, "universe_domain"):
            return
        if getattr(credentials_cls, "_universe_domain_patched", False):
            return

        # Store original __init__ method
        original_init = credentials_cls.__init__

        # Create patched init method that adds the attribute when missing
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, 'universe_domain'):
                self.universe_domain = 'googleapis.com'
        
        # Apply patch
        credentials_cls.__init__ = patched_init
        credentials_cls._universe_domain_patched = True
        logger.info("Applied patch for Google Cloud Storage credentials universe_domain attribute")
        
    except Exception as e:
        logger.warning(f"Failed to apply Google Cloud Storage credentials patch: {e}")

# Apply patch when module is loaded
patch_gcs_credentials()

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
            
            # Downcast to float16 to minimize disk usage while preserving shape
            embeddings_to_store = embeddings
            if embeddings.dtype != np.float16:
                embeddings_to_store = embeddings.astype(np.float16)
                logger.info("Downcasting embeddings to float16 for compact file storage")
            
            # Save embeddings
            np.savez_compressed(self.embeddings_path, embeddings=embeddings_to_store)
            
            # Save thread ID map
            with open(self.thread_id_map_path, 'w') as f:
                json.dump(thread_id_map, f)

            # Save embedding status for compatibility with downstream checks
            status_path = Path(self.config.stratified_data_path) / "embedding_status.csv"
            status_df = pd.DataFrame({
                "thread_id": list(thread_id_map.keys()),
                "has_embedding": [True] * len(thread_id_map)
            })
            status_df.to_csv(status_path, index=False)
            
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
            
            # Ensure embeddings are float32 for downstream compatibility
            if embeddings.dtype != np.float32:
                original_dtype = embeddings.dtype
                embeddings = embeddings.astype(np.float32)
                logger.info(f"Converted embeddings to float32 from {original_dtype}")
            
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

# Replit implementations (use our existing classes from replit.py when available)
if PostgresDB is None:
    class PostgresDB:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            if os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes"):
                self._data = pd.DataFrame()
            else:
                raise ImportError("PostgresDB unavailable; install replit/psycopg2 dependencies")

        def initialize_schema(self):
            return None

        def sync_data_from_dataframe(self, df: pd.DataFrame) -> int:
            if df is None:
                return 0
            if self._data.empty:
                self._data = df.copy()
            else:
                self._data = pd.concat([self._data, df.copy()], ignore_index=True)
            return len(df)

        def get_complete_data(self, filter_date: Optional[str] = None) -> pd.DataFrame:
            if self._data is None:
                return pd.DataFrame()
            return self._data.copy()

        def check_data_needs_update(self):
            return False, None

        def get_row_count(self) -> int:
            return len(self._data) if hasattr(self, "_data") and self._data is not None else 0

if KeyValueStore is None:
    class KeyValueStore:  # type: ignore[override]
        def __init__(self):
            self._state = {}

        def update_processing_state(self, state):
            self._state["processing_state"] = state

        def get_processing_state(self):
            return self._state.get("processing_state")

class ReplitCompleteDataStorage(CompleteDataStorage):
    """Replit PostgreSQL implementation of complete dataset storage."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        if PostgresDB is None:
            raise ImportError("PostgresDB unavailable; install replit/psycopg2 dependencies")
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
                            # Get source filename from DataFrame attributes if available
                            current_filename = chunk.attrs.get('source_filename', file_path)
                            # Store chunk in database with the source filename
                            await self.store_data(filtered_chunk, current_filename=current_filename)
                        
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
    
    async def store_data(self, df: pd.DataFrame, current_filename: Optional[str] = None) -> bool:
        """Store complete dataset in PostgreSQL database.
        
        Args:
            df: DataFrame containing data to sync
            current_filename: The actual source filename for this chunk
        """
        try:
            # Use the new sync method to only add new data
            rows_added = self.db.sync_data_from_dataframe(df)
            logger.info(f"Synchronized complete data with PostgreSQL, added {rows_added} new rows")
            
            # Use the passed filename if provided, otherwise use a default message
            file_ref = current_filename if current_filename else "unknown source"
            logger.info(f"Processed chunk with {len(df)} rows from file {file_ref}")
            
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
    
    def _init_object_client(self, bucket=None):
        """Initialize Object Storage client with proper error handling.
        
        Returns:
            Object Storage client instance or None if initialization fails
        """
        test_mode = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")
        if test_mode:
            return _InMemoryObjectClient()
        try:
            # Apply credential patch if needed before client initialization
            patch_gcs_credentials()
            
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                if bucket:
                    return Client(bucket=bucket)
                else:
                    return Client()
            except ValueError as bucket_error:
                # Handle no default bucket error
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}'")
                    return Client(bucket=self.default_bucket)
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to initialize Object Storage client: {e}")
            logger.error(traceback.format_exc())
            if test_mode:
                return _InMemoryObjectClient()
            return None
    
    async def store_sample(self, df: pd.DataFrame) -> bool:
        """Store stratified sample to Object Storage."""
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return False
        
        try:
            # Initialize client with patched method
            object_client = self._init_object_client()
            
            if not object_client:
                logger.error("Failed to initialize Object Storage client")
                return False
            
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
            # Initialize client with patched method
            object_client = self._init_object_client()
            
            # If client initialization failed, try fallback to key-value store
            if not object_client:
                logger.warning("Failed to initialize Object Storage client, trying key-value store fallback")
                from config.replit import KeyValueStore
                kv_store = KeyValueStore()
                sample = kv_store.get_stratified_sample()
                if sample is not None:
                    logger.info(f"Retrieved stratified sample with {len(sample)} rows from key-value store")
                    return sample
                logger.warning("No sample found in key-value store, returning empty DataFrame")
                return pd.DataFrame()
            
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
            
            # Try falling back to key-value store
            logger.warning("Trying key-value store fallback")
            try:
                from config.replit import KeyValueStore
                kv_store = KeyValueStore()
                sample = kv_store.get_stratified_sample()
                if sample is not None:
                    logger.info(f"Retrieved stratified sample with {len(sample)} rows from key-value store")
                    return sample
            except Exception as kv_error:
                logger.error(f"Failed to get sample from key-value store: {kv_error}")
            
            return None
    
    async def sample_exists(self) -> bool:
        """Check if stratified sample exists in Object Storage."""
        try:
            # Initialize client with patched method
            object_client = self._init_object_client()
            
            if not object_client:
                logger.warning("Failed to initialize Object Storage client, assuming sample exists")
                return True
            
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
            # In deployed environment, assume the sample exists to prevent unnecessary refreshes
            logger.warning("Error checking if sample exists, assuming it does exist to prevent unnecessary refreshes")
            return True

class ReplitObjectEmbeddingStorage(EmbeddingStorage):
    """Object Storage-based implementation of embedding storage for Replit."""
    
    def __init__(self, config):
        """Initialize Object Storage client."""
        self.config = config
        self.embeddings_key = "embeddings.npy"
        self.thread_map_key = "thread_id_map.json"
        self.replit_kv = None  # For state tracking only
        self.default_bucket = "knowledge-agent-embeddings"  # Default bucket name
    
    def _init_object_client(self, bucket=None):
        """Initialize Object Storage client with proper error handling.
        
        Returns:
            Object Storage client instance or None if initialization fails
        """
        test_mode = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")
        if test_mode:
            return _InMemoryObjectClient()
        try:
            # Apply credential patch if needed before client initialization
            patch_gcs_credentials()
            
            # Import Object Storage client
            from replit.object_storage import Client
            
            try:
                if bucket:
                    return Client(bucket=bucket)
                else:
                    return Client()
            except ValueError as bucket_error:
                # Handle no default bucket error
                if "no default bucket" in str(bucket_error).lower():
                    logger.warning(f"No default bucket configured. Using '{self.default_bucket}'")
                    return Client(bucket=self.default_bucket)
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to initialize Object Storage client: {e}")
            logger.error(traceback.format_exc())
            if test_mode:
                return _InMemoryObjectClient()
            return None
    
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
            
            embeddings_to_store = embeddings
            if embeddings.dtype != np.float16:
                embeddings_to_store = embeddings.astype(np.float16)
                logger.info("Downcasting embeddings to float16 before uploading to Object Storage")
            
            # Initialize client with patched method
            object_client = self._init_object_client()
            
            if not object_client:
                logger.error("Failed to initialize Object Storage client")
                return False
            
            # Ensure consistent thread ID format (convert all to strings)
            clean_thread_map = {str(k).strip(): v for k, v in thread_id_map.items()}
            
            # Add metadata to thread map
            thread_map_with_meta = {
                'thread_ids': clean_thread_map,
                'embedding_shape': embeddings.shape,
                'embedding_dtype': str(embeddings_to_store.dtype),
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
                np.save(temp_file, embeddings_to_store)
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
            
            # Initialize client with patched method
            object_client = self._init_object_client()
            
            if not object_client:
                logger.error("Failed to initialize Object Storage client")
                return None, None
            
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
            
            if embeddings is not None and embeddings.dtype != np.float32:
                original_dtype = embeddings.dtype
                embeddings = embeddings.astype(np.float32)
                logger.info(f"Converted embeddings to float32 from {original_dtype}")
            
            logger.info(f"Retrieved embeddings with shape {embeddings.shape} from Object Storage")
            return embeddings, thread_id_map
        except Exception as e:
            logger.error(f"Error retrieving embeddings from Object Storage: {e}")
            logger.error(traceback.format_exc())
            return None, None
    
    async def embeddings_exist(self) -> bool:
        """Check if embeddings exist in Object Storage."""
        try:
            # Initialize client with patched method
            object_client = self._init_object_client()
            
            if not object_client:
                logger.error("Failed to initialize Object Storage client")
                return False
            
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
        test_mode = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")
        if test_mode:
            logger.info("TEST_MODE enabled; using in-memory state store for ReplitStateManager")
            self.kv_store = _InMemoryStateStore()
            return

        if KeyValueStore is None:
            raise ImportError("KeyValueStore unavailable; install replit dependencies")

        try:
            self.kv_store = KeyValueStore()
        except Exception as e:
            logger.error("Failed to initialize KeyValueStore: %s", e)
            raise
    
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

# Backwards-compatible alias for tests and external imports
class ReplitEmbeddingStorage(ReplitObjectEmbeddingStorage):
    pass

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
            env_type = getattr(config, "env", None) or detect_environment()
        
        logger.info(f"Creating storage implementations for environment: {env_type}")
        
        if env_type.lower() == 'replit':
            return {
                'complete_data': ReplitCompleteDataStorage(config),
                'stratified_sample': ReplitStratifiedSampleStorage(config),
                'embeddings': ReplitEmbeddingStorage(config),
                'state': ReplitStateManager(config)
            }
        else:
            return {
                'complete_data': FileCompleteDataStorage(config),
                'stratified_sample': FileStratifiedSampleStorage(config),
                'embeddings': FileEmbeddingStorage(config),
                'state': FileStateManager(config)
            }
