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

from config.env_loader import detect_environment

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
    """Replit Key-Value store implementation of stratified sample storage."""
    
    def __init__(self, config):
        self.config = config
        self.kv_store = KeyValueStore()
    
    async def store_sample(self, df: pd.DataFrame) -> bool:
        """Store stratified sample to Key-Value store."""
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return False
        
        try:
            self.kv_store.store_stratified_sample(df)
            logger.info(f"Stored stratified sample with {len(df)} rows to Key-Value store")
            return True
        except Exception as e:
            logger.error(f"Error storing stratified sample: {e}")
            return False
    
    async def get_sample(self) -> Optional[pd.DataFrame]:
        """Retrieve stratified sample from Key-Value store."""
        try:
            return self.kv_store.get_stratified_sample()
        except Exception as e:
            logger.error(f"Error retrieving stratified sample: {e}")
            return None
    
    async def sample_exists(self) -> bool:
        """Check if stratified sample exists in Key-Value store."""
        try:
            # Try to get the sample metadata
            meta_key = f"{self.kv_store.STRATIFIED_SAMPLE_KEY}_meta"
            from replit import db as kv_db
            return meta_key in kv_db
        except Exception as e:
            logger.error(f"Error checking if sample exists: {e}")
            return False

class ReplitEmbeddingStorage(EmbeddingStorage):
    """Replit Key-Value store implementation of embedding storage."""
    
    def __init__(self, config):
        self.config = config
        self.kv_store = KeyValueStore()
    
    async def store_embeddings(self, embeddings: np.ndarray, thread_id_map: Dict[str, int]) -> bool:
        """Store embeddings to Key-Value store."""
        if embeddings.size == 0 or not thread_id_map:
            logger.warning("Empty embeddings or thread ID map, nothing to store")
            return False
        
        try:
            self.kv_store.store_embeddings(embeddings, thread_id_map)
            logger.info(f"Stored embeddings with shape {embeddings.shape} and {len(thread_id_map)} thread IDs to Key-Value store")
            return True
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    async def get_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Retrieve embeddings from Key-Value store."""
        try:
            return self.kv_store.get_embeddings()
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return None, None
    
    async def embeddings_exist(self) -> bool:
        """Check if embeddings exist in Key-Value store."""
        try:
            # Try to get the embeddings metadata
            meta_key = f"{self.kv_store.EMBEDDINGS_KEY}_meta"
            from replit import db as kv_db
            return meta_key in kv_db
        except Exception as e:
            logger.error(f"Error checking if embeddings exist: {e}")
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