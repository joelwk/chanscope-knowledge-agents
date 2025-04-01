import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import pytz
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Awaitable
from filelock import FileLock, Timeout

import numpy as np
import pandas as pd

from config.base_settings import get_base_settings
from config.settings import Config
from .data_processing.cloud_handler import load_all_csv_data_from_s3, S3Handler
from .data_processing.sampler import Sampler
from .embedding_ops import get_relevant_content, load_embeddings, load_thread_id_map
from config.env_loader import detect_environment

# Initialize logger
logger = logging.getLogger(__name__)

# Define utility function for parsing dates
def parse_filter_date(date_str: str) -> str:
    """Parse a date string into a standardized format."""
    try:
        # Convert to datetime object for validation
        if isinstance(date_str, (datetime, pd.Timestamp)):
            dt = date_str
        else:
            dt = pd.to_datetime(date_str)
        
        # Return formatted string
        return dt.strftime('%Y-%m-%d %H:%M:%S%z')
    except Exception as e:
        logger.error(f"Error parsing date {date_str}: {e}")
        raise ValueError(f"Invalid date format: {date_str}")

@dataclass
class DataConfig:
    """Configuration for data operations."""
    root_data_path: Path
    stratified_data_path: Optional[Path] = None
    temp_path: Optional[Path] = None
    filter_date: Optional[str] = None
    sample_size: int = 1000
    time_column: str = 'posted_date_time'
    strata_column: str = 'thread_id'
    board_id: Optional[str] = None
    force_refresh: bool = False

    def __post_init__(self):
        chunk_settings = Config.get_chunk_settings()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()
        paths = Config.get_paths()

        self.dtype_optimizations = column_settings['column_types']
        self.time_formats = ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S']
        self.processing_chunk_size = chunk_settings['processing_chunk_size']
        self.stratification_chunk_size = chunk_settings['stratification_chunk_size']

        self.sample_size = min(max(sample_settings['min_sample_size'], sample_settings['default_sample_size']),
                               sample_settings['max_sample_size'])
        if self.sample_size != sample_settings['default_sample_size']:
            logger.warning(f"Sample size adjusted to {self.sample_size}")

        self.root_data_path = Path(paths['root_data_path'])
        self.stratified_data_path = Path(paths['stratified'])
        self.temp_path = Path(paths['temp'])

        if self.filter_date:
            try:
                self.filter_date = parse_filter_date(self.filter_date)
            except ValueError as e:
                logger.error("Error parsing filter_date", exc_info=True)
                self.filter_date = None

    @property
    def read_csv_kwargs(self) -> Dict[str, Any]:
        return {
            'dtype': self.dtype_optimizations,
            'on_bad_lines': 'warn',
            'low_memory': True,
            'parse_dates': [self.time_column],
            'date_format': self.time_formats[0]
        }

    @classmethod
    def from_config(cls, force_refresh: bool = False) -> 'DataConfig':
        base_settings = get_base_settings()
        paths = base_settings.get('paths', {})
        processing = base_settings.get('processing', {})
        sample_settings = base_settings.get('sample', {})
        columns = base_settings.get('columns', {})

        # Get filter_date from processing settings
        filter_date = processing.get('filter_date')
        logger.info(f"Using filter_date from settings: {filter_date}")
        
        # Get sample size from sample settings
        sample_size = sample_settings.get('default_sample_size', 1000)
        
        # Get time column and strata column from column settings
        time_column = columns.get('time_column', 'posted_date_time')
        strata_column = columns.get('strata_column')

        root_data_path = paths.get('root_data_path', 'data')
        stratified_path = paths.get('stratified', os.path.join(root_data_path, 'stratified'))
        temp_path = paths.get('temp', 'temp_files')

        return cls(
            root_data_path=root_data_path,
            stratified_data_path=stratified_path,
            temp_path=temp_path,
            filter_date=filter_date,
            sample_size=sample_size,
            time_column=time_column,
            strata_column=strata_column,
            force_refresh=force_refresh
        )


class DataStateManager:
    """Manages state and validation of data files."""
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._state_file = self.config.temp_path / '.data_state'
        self._ensure_state_file()

    def _ensure_state_file(self) -> None:
        if not self._state_file.exists():
            self._save_state({'last_update': None, 'total_records': 0})

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            if state.get('last_update') is not None:
                if isinstance(state['last_update'], (pd.Timestamp, datetime)):
                    state['last_update'] = state['last_update'].isoformat()
            state['total_records'] = int(state.get('total_records', 0))
            with open(self._state_file, 'w') as f:
                json.dump(state, f)
            self._logger.info(f"State saved: {state}")
        except Exception as e:
            self._logger.exception("Error saving state")
            raise

    def get_last_update(self) -> Optional[pd.Timestamp]:
        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)
            last_update = state.get('last_update')
            return pd.to_datetime(last_update, utc=True) if last_update else None
        except Exception as e:
            self._logger.exception("Error reading state")
            return None

    def update_state(self, total_records: int) -> None:
        state = {'last_update': pd.Timestamp.now(tz='UTC'), 'total_records': total_records}
        self._save_state(state)
        self._logger.info(f"State updated: {state}")

    def validate_file_structure(self) -> Dict[str, bool]:
        results = {}
        paths = {
            'root_data_path': self.config.root_data_path,
            'stratified_data_path': self.config.stratified_data_path,
            'temp_path': self.config.temp_path,
            'complete_data': self.config.root_data_path / 'complete_data.csv'
        }
        for name, path in paths.items():
            try:
                exists = path.exists() and (path.is_dir() if name != 'complete_data' else path.parent.exists())
                if not exists:
                    self._logger.warning(f"Validation failed for {name}: {path}")
                results[name] = exists
            except Exception as e:
                self._logger.exception(f"Validation error for {name}")
                results[name] = False
        return results

    def validate_data_integrity(self) -> Dict[str, bool]:
        integrity = {'complete_data_valid': False, 'stratified_data_valid': False, 'embeddings_valid': False}
        try:
            complete_data_path = self.config.root_data_path / 'complete_data.csv'
            if complete_data_path.exists():
                df = pd.read_csv(complete_data_path)
                integrity['complete_data_valid'] = not df.empty
                self._logger.info(f"Complete data rows: {len(df)}")

            stratified_file = self.config.stratified_data_path / 'stratified_sample.csv'
            if stratified_file.exists():
                df = pd.read_csv(stratified_file)
                integrity['stratified_data_valid'] = not df.empty
                self._logger.info(f"Stratified data rows: {len(df)}")

            embeddings_path = self.config.stratified_data_path / 'embeddings.npz'
            thread_id_map_path = self.config.stratified_data_path / 'thread_id_map.json'
            if embeddings_path.exists() and thread_id_map_path.exists():
                with np.load(embeddings_path) as data:
                    embeddings = data.get('embeddings')
                with open(thread_id_map_path, 'r') as f:
                    thread_id_map = json.load(f)
                integrity['embeddings_valid'] = bool(embeddings is not None and len(embeddings) and len(thread_id_map))
                self._logger.info(f"Embeddings count: {len(embeddings) if embeddings is not None else 0}")
            else:
                self._logger.warning("Embeddings or thread_id mapping missing")
            return integrity
        except Exception as e:
            self._logger.exception("Error validating data integrity")
            return integrity

    def check_data_integrity(self) -> bool:
        file_structure = self.validate_file_structure()
        data_integrity = self.validate_data_integrity()
        all_valid = {**file_structure, **data_integrity}
        failed = [name for name, valid in all_valid.items() if not valid]
        if failed:
            self._logger.warning(f"Data integrity check failed: {', '.join(failed)}")
            return False
        return True

    def verify_data_structure(self) -> Dict[str, bool]:
        required_files = {
            'complete_data': self.config.root_data_path / 'complete_data.csv',
            'stratified_dir': self.config.stratified_data_path,
            'stratified_sample': self.config.stratified_data_path / 'stratified_sample.csv'
        }
        status = {}
        for name, path in required_files.items():
            exists = path.exists()
            self._logger.info(f"Checking {name}: {'✓' if exists else '✗'} ({path})")
            status[name] = exists
        return status


class DataProcessor:
    """Handles core data processing operations including stratification."""
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self.chunk_size = config.processing_chunk_size
        self.sampler = Sampler(
            filter_date=config.filter_date,
            time_column=config.time_column,
            strata_column=config.strata_column,
            initial_sample_size=config.sample_size
        )
        self.required_columns = {
            'thread_id': str,
            'posted_date_time': str,
            'text_clean': str,
            'posted_comment': str
        }

    async def stratify_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self._logger.info(f"Stratifying data with {len(data)} records")
        missing = set(self.required_columns.keys()) - set(data.columns)
        if missing:
            self._logger.error(f"Missing columns: {missing}. Available: {data.columns.tolist()}")
            raise ValueError(f"Missing columns: {missing}")
        stratified = self.sampler.stratified_sample(data)
        self._logger.info(f"Stratification complete; result size: {len(stratified)}")
        return stratified

    async def _update_stratified_sample(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load and process stratified sample data from storage."""
        try:
            # Define paths - use self.config.stratified_data_path for consistency
            stratified_file = self.config.stratified_data_path / 'stratified_sample.csv'
            
            # Check if stratified sample already exists and we're not forcing refresh
            if not force_refresh and stratified_file.exists():
                try:
                    df = pd.read_csv(stratified_file)
                    if not df.empty:
                        logger.info(f"Loaded existing stratified sample with {len(df)} records")
                        return df
                except Exception as e:
                    logger.warning(f"Error loading existing stratified sample: {e}, regenerating")
                    
            # Sample data doesn't exist or we're forcing refresh, generate it
            logger.info("Generating new stratified sample")
            
            # For now, we'll use a simplified approach to create a sample dataset
            # In a real-world scenario, you'd likely load this from a database or API
            
            # Create a basic dataframe with sample data
            sample_data = []
            for i in range(100):
                sample_data.append({
                    'thread_id': f'thread_{i}',
                    'posted_date_time': datetime.now(pytz.UTC).isoformat(),
                    'text_clean': f'This is sample text for article {i}. It contains information about a topic of interest.'
                })
                
            df = pd.DataFrame(sample_data)
            
            # Save to CSV - ensure parent directory exists
            self.config.stratified_data_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(stratified_file, index=False)
            logger.info(f"Generated and saved stratified sample with {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error updating stratified sample: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def _mark_update_complete(self):
        """Mark the update as complete by creating a flag file."""
        flag_path = self.config.root_data_path / ".update_complete"
        flag_path.touch()


class DataOperations:
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self.processor = DataProcessor(config)
        self.state_manager = DataStateManager(config)
        self.retention_days = Config.get_processing_settings().get('retention_days', 30)
        self.read_csv_kwargs = {'dtype': config.dtype_optimizations, 'parse_dates': [config.time_column]}
        self._embedding_lock = asyncio.Lock()
        self._embedding_update_lock = asyncio.Lock()
        self._is_embedding_updating = False
        self._data_loading_lock = asyncio.Lock()
        self._stratified_lock = asyncio.Lock()
        self._cached_chunks = {}
        
        # File-based locking setup
        self._lock_file_path = self.config.temp_path / "data_update.lock"
        self._update_flag_file = self.config.temp_path / 'data_update_complete.json'
        self._in_progress_flag_file = self.config.temp_path / 'data_update_inprogress.json'
        self._lock_timeout = 300  # 5 minutes timeout for lock acquisition

    def _acquire_lock(self) -> Optional[FileLock]:
        """Attempt to acquire the file lock, clearing stale locks if necessary."""
        try:
            # Check if a stale lock file exists and remove it if it's older than the timeout
            if self._lock_file_path.exists():
                lock_age = time.time() - self._lock_file_path.stat().st_mtime
                if lock_age > self._lock_timeout:
                    self._logger.warning(f"Stale lock detected (age {lock_age:.2f}s), removing it.")
                    self._lock_file_path.unlink()
            lock = FileLock(str(self._lock_file_path), timeout=self._lock_timeout)
            lock.acquire(timeout=0)  # Non-blocking attempt
            self._logger.info("Acquired file lock successfully.")
            return lock
        except Timeout:
            self._logger.info("Another process holds the lock; skipping update")
            return None
        except Exception as e:
            self._logger.warning(f"Error acquiring lock: {e}")
            return None

    def _release_lock(self, lock: Optional[FileLock]) -> None:
        """Safely release the file lock."""
        if lock is not None:
            try:
                lock.release()
            except Exception as e:
                self._logger.warning(f"Error releasing lock: {e}")

    def _is_update_recent(self) -> bool:
        """Check if the data has been updated recently."""
        if not self._update_flag_file.exists():
            return False
        
        try:
            with open(self._update_flag_file, 'r') as f:
                update_info = json.load(f)
                last_update = datetime.fromisoformat(update_info.get('timestamp', '2000-01-01T00:00:00'))
                # Ensure last_update has timezone info if it doesn't already
                if last_update.tzinfo is None:
                    last_update = last_update.replace(tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - last_update).total_seconds() < 3600  # 1 hour
        except Exception as e:
            logger.warning(f"Failed to check update recency: {e}")
            return False

    async def check_data_freshness(self) -> Dict[str, Any]:
        """Check if the data is fresh and ready for use.
        
        Returns:
            Dict with status information about data freshness
        """
        result = {
            "is_fresh": False,
            "last_update": None,
            "needs_refresh": True
        }
        
        # Check if update is recent
        if self._is_update_recent():
            with open(self._update_flag_file, 'r') as f:
                try:
                    update_info = json.load(f)
                    result["is_fresh"] = True
                    result["last_update"] = update_info.get('timestamp')
                    result["needs_refresh"] = False
                except (json.JSONDecodeError, KeyError):
                    pass
                    
        # Check data state from state manager
        last_update = self.state_manager.get_last_update()
        if last_update and not result["last_update"]:
            result["last_update"] = last_update.isoformat()
            
        # Check if files exist and have content
        data_integrity = self.state_manager.validate_data_integrity()
        result["file_status"] = data_integrity
        
        if all(data_integrity.values()) and not result["needs_refresh"]:
            result["is_fresh"] = True
            
        return result
        
    def _mark_update_complete(self) -> None:
        """Mark the data update as complete in the flag file."""
        try:
            self._update_flag_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._update_flag_file, "w") as f:
                json.dump({
                    "last_update": pd.Timestamp.now(tz='UTC').isoformat(),
                    "status": "complete"
                }, f)
            self._logger.info("Marked data update as complete (persistent flag)")
        except Exception as e:
            self._logger.warning(f"Error writing update flag: {e}")

    async def _update_embeddings(
        self,
        force_refresh: bool = False,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], Union[None, Awaitable[None]]]] = None
    ) -> None:
        """Update embeddings for the stratified dataset."""
        try:
            # Initialize Object Storage
            from config.storage import ReplitObjectEmbeddingStorage
            embedding_storage = ReplitObjectEmbeddingStorage(self.config)
            
            # Check if we need to update embeddings
            if not force_refresh:
                embeddings_exist = await embedding_storage.embeddings_exist()
                if embeddings_exist:
                    logger.info("Embeddings already exist and force_refresh is False, skipping update")
                    return
            
            # Load stratified data
            stratified_data = await self._load_stratified_data()
            if stratified_data is None or len(stratified_data) == 0:
                logger.error("No stratified data available for embedding generation")
                return
            
            # Get thread IDs and content for embedding generation
            thread_ids = stratified_data['thread_id'].astype(str).tolist()
            content = stratified_data['text_clean'].tolist()
            
            if not thread_ids or not content:
                logger.error("No content or thread IDs available for embedding generation")
                return
            
            # Process in batches to avoid memory issues
            batch_size = 100
            all_embeddings = []
            thread_id_map = {}
            
            for i in range(0, len(content), batch_size):
                batch_text = content[i:i+batch_size]
                batch_ids = thread_ids[i:i+batch_size]
                
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{len(content)//batch_size + 1}")
                
                # Generate embeddings for batch
                batch_embeddings = await self.embedding_provider.get_embeddings(batch_text)
                
                if batch_embeddings is None:
                    logger.error("Failed to generate batch embeddings")
                    return
                
                # Add to results
                for j, embedding in enumerate(batch_embeddings):
                    idx = len(all_embeddings)
                    all_embeddings.append(embedding)
                    thread_id_map[batch_ids[j]] = idx
                
                logger.info(f"Completed batch with {len(batch_embeddings)} embeddings")
                await asyncio.sleep(0.1)  # Small delay to prevent API rate limits
            
            # Convert to numpy array for storage
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Store embeddings in Object Storage
            success = await embedding_storage.store_embeddings(embeddings_array, thread_id_map)
            
            if success:
                logger.info(f"Successfully generated and stored {len(all_embeddings)} embeddings")
            else:
                logger.error("Failed to store embeddings in Object Storage")
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            logger.error(traceback.format_exc())

    async def ensure_data_ready(
        self,
        force_refresh: bool = False,
        skip_embeddings: bool = False
    ) -> bool:
        """
        Ensures all necessary data is ready for inference.
        
        Args:
            force_refresh: If True, forces a refresh of all data regardless of current state
            skip_embeddings: If True, skips embedding generation step
            
        Returns:
            bool: True if data is ready, False otherwise
        """
        logger.info(f"Ensuring data is ready (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})")
        
        try:
            # Initialize storage implementations
            from config.storage import StorageFactory
            storage = StorageFactory.create(self.config)
            
            complete_data_storage = storage['complete_data']
            stratified_storage = storage['stratified_sample']
            embedding_storage = storage['embeddings']
            
            # Check if we need to update the database
            if force_refresh or await self.needs_complete_update():
                logger.info("Preparing complete dataset...")
                
                try:
                    # Download and process data
                    start_time = time.time()
                    success = await complete_data_storage.prepare_data()
                    
                    if success:
                        logger.info(f"Complete dataset prepared in {time.time() - start_time:.2f}s")
                    else:
                        logger.error("Failed to prepare complete dataset")
                        return False
                except Exception as e:
                    logger.error(f"Error preparing complete dataset: {str(e)}")
                    logger.error(traceback.format_exc())
                    return False
            
            # Check if we need to update the stratified sample
            if force_refresh or await self.needs_stratification():
                logger.info("Creating stratified sample...")
                
                try:
                    # Get complete data for stratification
                    complete_data = await complete_data_storage.get_data()
                    if complete_data.empty:
                        logger.error("No complete data available for stratification")
                        return False
                    
                    # Create stratified sample
                    start_time = time.time()
                    stratified_data = await self.processor.stratify_data(complete_data)
                    
                    if not stratified_data.empty:
                        # Store stratified sample
                        success = await stratified_storage.store_sample(stratified_data)
                        if success:
                            logger.info(f"Stratified sample created and stored in {time.time() - start_time:.2f}s")
                        else:
                            logger.error("Failed to store stratified sample")
                            return False
                    else:
                        logger.error("Failed to create stratified sample")
                        return False
                except Exception as e:
                    logger.error(f"Error creating stratified sample: {str(e)}")
                    logger.error(traceback.format_exc())
                    return False
            
            # Check if we need to update embeddings
            if not skip_embeddings and (force_refresh or await self._needs_embeddings_update()):
                logger.info("Generating embeddings...")
                
                try:
                    # Load stratified data for embedding generation
                    stratified_data = await stratified_storage.get_sample()
                    if stratified_data is None or stratified_data.empty:
                        logger.error("No stratified data available for embedding generation")
                        return False
                    
                    # Generate embeddings
                    start_time = time.time()
                    await self._update_embeddings(force_refresh=force_refresh)
                    
                    # Verify embeddings were generated
                    embeddings_exist = await embedding_storage.embeddings_exist()
                    if embeddings_exist:
                        logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s")
                    else:
                        logger.error("Failed to generate embeddings")
                        if not skip_embeddings:
                            return False
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    logger.error(traceback.format_exc())
                    if not skip_embeddings:
                        return False
            
            # Data is ready
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in ensure_data_ready: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def _ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        await asyncio.gather(
            self.config.root_data_path.mkdir(parents=True, exist_ok=True),
            self.config.stratified_data_path.mkdir(parents=True, exist_ok=True),
            self.config.temp_path.mkdir(parents=True, exist_ok=True)
        )

    async def _download_and_process_data(self) -> bool:
        """Download data from S3 and process it."""
        try:
            # Calculate date range based on filter_date
            start_time = None
            if self.config.filter_date:
                try:
                    start_time = pd.to_datetime(self.config.filter_date)
                except Exception as e:
                    logger.warning(f"Error parsing filter_date '{self.config.filter_date}': {e}")
                    start_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=14)
            else:
                # Default to last 14 days
                start_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=14)
                
            # Load complete dataset
            complete_data_path = self.config.root_data_path / 'complete_data.csv'
            
            # Use either S3 data or mock data if in test mode
            if self.config.test_mode:
                logger.info("Using mock data in test mode")
                data_df = self._create_mock_data()
            else:
                logger.info(f"Fetching data from S3 starting from {start_time}")
                # Import here to avoid issues if S3 modules aren't available
                from .data_processing.cloud_handler import S3Handler, load_all_csv_data_from_s3
                
                # Check S3 connectivity
                s3_handler = S3Handler()
                if not s3_handler.is_configured:
                    logger.error("S3 is not properly configured")
                    return False
                
                # Stream data from S3
                complete_df = []
                record_count = 0
                
                data_generator = load_all_csv_data_from_s3(
                    latest_date_processed=start_time.isoformat(),
                    chunk_size=self.config.processing_chunk_size,
                    board_id=self.config.board_id
                )
                
                async for chunk in data_generator:
                    # Process date column
                    chunk[self.config.time_column] = pd.to_datetime(
                        chunk[self.config.time_column], 
                        format='mixed',
                        utc=True,
                        errors='coerce'
                    )
                    
                    # Filter by date
                    date_mask = chunk[self.config.time_column] >= start_time
                    chunk = chunk[date_mask]
                    
                    record_count += len(chunk)
                    complete_df.append(chunk)
                    logger.info(f"Processed chunk with {len(chunk)} rows (Total: {record_count})")
                    
                    # Yield to other tasks
                    await asyncio.sleep(0)
                
                # Combine all chunks
                if not complete_df:
                    logger.error("No data fetched from S3")
                    return False
                
                data_df = pd.concat(complete_df, ignore_index=True)
            
            # Save to CSV
            complete_data_path.parent.mkdir(parents=True, exist_ok=True)
            data_df.to_csv(complete_data_path, index=False)
            logger.info(f"Saved {len(data_df)} rows to {complete_data_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading and processing data: {e}")
            logger.error(traceback.format_exc())
            return False

    async def _needs_embeddings_update(self) -> bool:
        """Check if embeddings need to be updated."""
        try:
            embeddings_path = self.config.stratified_data_path / 'embeddings.npz'
            thread_id_map_path = self.config.stratified_data_path / 'thread_id_map.json'
            
            # Check if embedding files exist and are valid
            if not embeddings_path.exists() or not thread_id_map_path.exists():
                logger.info("Embeddings files do not exist, update needed")
                return True
            
            # Check if embedding files are valid
            try:
                # Try to load embedding files
                np.load(embeddings_path)
                with open(thread_id_map_path, 'r') as f:
                    thread_map = json.load(f)
                
                # Check if thread map is valid
                if not isinstance(thread_map, dict) or len(thread_map) == 0:
                    logger.info("Thread map is invalid, update needed")
                    return True
                
                logger.info(f"Embeddings are valid with {len(thread_map)} entries")
                return False
            except Exception as e:
                logger.warning(f"Error validating embedding files: {e}")
                return True
            
        except Exception as e:
            logger.error(f"Error checking if embeddings need update: {e}")
            logger.error(traceback.format_exc())
            return True

    async def _create_stratified_sample(self) -> pd.DataFrame:
        """Create a stratified sample from the given data."""
        self._logger.info("Creating stratified sample")
        stratified = self.processor.stratify_data(await self._load_stratified_data())
        self._logger.info(f"Stratification complete; result size: {len(stratified)}")
        return stratified

    async def _generate_embeddings(self) -> bool:
        """
        Generate embeddings for stratified data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Generating embeddings for stratified data")
            
            # Load stratified data if not already loaded
            stratified_file = self.config.stratified_data_path / 'stratified_sample.csv'
            
            if not stratified_file.exists():
                logger.error(f"Stratified data file not found: {stratified_file}")
                return False
                
            # Load stratified data
            stratified_data = pd.read_csv(stratified_file)
            logger.info(f"Loaded {len(stratified_data)} rows from {stratified_file}")
            
            # Initialize embedding provider
            from .embedding import EmbeddingProvider
            embedding_provider = EmbeddingProvider()
            
            # Ensure we have the necessary text field for embedding
            text_column = 'text_clean'
            if text_column not in stratified_data.columns:
                if 'content' in stratified_data.columns:
                    text_column = 'content'
                    logger.info(f"Using '{text_column}' column for embeddings")
                else:
                    logger.error("No suitable text column found for embeddings")
                    return False
            
            # Check if we have thread_id column
            if 'thread_id' not in stratified_data.columns:
                logger.error("No thread_id column found in stratified data")
                return False
            
            # Get text content to embed
            text_content = stratified_data[text_column].values
            thread_ids = stratified_data['thread_id'].astype(str).values
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(text_content)} threads")
            
            # Process in batches to avoid memory issues
            batch_size = 100
            all_embeddings = []
            thread_id_map = {}
            
            for i in range(0, len(text_content), batch_size):
                batch_text = text_content[i:i+batch_size]
                batch_ids = thread_ids[i:i+batch_size]
                
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{len(text_content)//batch_size + 1}")
                
                # Generate embeddings for batch
                batch_embeddings = await embedding_provider.get_embeddings(batch_text)
                
                if batch_embeddings is None:
                    logger.error("Failed to generate batch embeddings")
                    return False
                
                # Add to results
                for j, embedding in enumerate(batch_embeddings):
                    idx = len(all_embeddings)
                    all_embeddings.append(embedding)
                    thread_id_map[batch_ids[j]] = idx
                
                logger.info(f"Completed batch with {len(batch_embeddings)} embeddings")
                await asyncio.sleep(0.1)  # Small delay to prevent API rate limits
            
            # Save embeddings
            embeddings_path = self.config.stratified_data_path / 'embeddings.npz'
            thread_id_map_path = self.config.stratified_data_path / 'thread_id_map.json'
            
            # Convert to numpy array for storage
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Save embeddings to file
            np.savez_compressed(embeddings_path, embeddings=embeddings_array)
            
            # Save thread ID map
            with open(thread_id_map_path, 'w') as f:
                json.dump(thread_id_map, f)
            
            logger.info(f"Successfully generated and saved {len(all_embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.error(traceback.format_exc())
            return False

    async def _cleanup_worker_markers(self):
        """Clean up old worker markers in the data directory."""
        logger.info("Cleaning up old worker markers")
        try:
            run_dir = Path(self.config.root_data_path)
            if not run_dir.exists():
                return

            # Cleanup worker markers older than 60 minutes
            cutoff_time = datetime.now(pytz.UTC) - timedelta(minutes=60)
            
            for worker_file in run_dir.glob(".worker_*_in_progress"):
                try:
                    file_mtime = datetime.fromtimestamp(worker_file.stat().st_mtime, tz=pytz.UTC)
                    if file_mtime < cutoff_time:
                        logger.info(f"Removing stale worker marker: {worker_file} (age: {(datetime.now(pytz.UTC) - file_mtime).total_seconds() / 60:.1f} min)")
                        worker_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error checking worker marker {worker_file}: {e}")
            
            # Check if main in-progress marker is stale
            in_progress_marker = run_dir / ".initialization_in_progress"
            if in_progress_marker.exists():
                try:
                    file_mtime = datetime.fromtimestamp(in_progress_marker.stat().st_mtime, tz=pytz.UTC)
                    if file_mtime < cutoff_time:
                        logger.info(f"Removing stale in-progress marker (age: {(datetime.now(pytz.UTC) - file_mtime).total_seconds() / 60:.1f} min)")
                        in_progress_marker.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error checking in-progress marker: {e}")
        
        except Exception as e:
            logger.warning(f"Error cleaning up worker markers: {e}")

    async def needs_complete_update(self, force_refresh: bool = False) -> bool:
        """Check if the complete dataset needs to be updated.
        
        Args:
            force_refresh: Whether to force refresh regardless of current state
            
        Returns:
            bool: True if update is needed, False otherwise
        """
        # Check environment type
        env_type = detect_environment()
        
        if env_type.lower() == 'replit':
            # For Replit, use storage implementation
            from config.storage import StorageFactory
            storage = StorageFactory.create(self.config, 'replit')
            complete_data_storage = storage['complete_data']
            
            try:
                # Check if data exists and has records
                row_count = await complete_data_storage.get_row_count()
                if row_count == 0:
                    self._logger.info("Complete data missing or empty in database")
                    return True
                
                # If forcing refresh, always update
                if force_refresh:
                    self._logger.info("Force refresh requested, complete data will be updated")
                    return True
                
                # Check if data is fresh
                is_fresh = await complete_data_storage.is_data_fresh()
                if not is_fresh:
                    self._logger.info("Complete data is not fresh, update needed")
                    return True
                
                self._logger.info(f"Complete data exists with {row_count} rows and is fresh")
                return False
            except Exception as e:
                self._logger.error(f"Error checking complete data in Replit storage: {e}")
                return True
        else:
            # For file-based storage
            complete_data_path = self.config.root_data_path / 'complete_data.csv'
            
            # Check if file exists and is valid
            if not complete_data_path.exists() or not self._verify_file(complete_data_path):
                self._logger.info("Complete dataset missing or invalid")
                return True
                
            # If forcing refresh, check if data is up-to-date with S3
            if force_refresh:
                self._logger.info("Force refresh requested, checking S3 status first")
                # Default to needing update for now
                return True
                
            # Try to read the file to check if it has data
            try:
                complete_data = pd.read_csv(complete_data_path)
                if complete_data.empty:
                    self._logger.info("Complete dataset is empty")
                    return True
                self._logger.info("Complete dataset exists and is not empty")
                return False
            except Exception:
                self._logger.exception("Error checking complete dataset")
                return True

    async def _load_stratified_data(self) -> pd.DataFrame:
        """Load stratified data and merge with embeddings from Object Storage."""
        try:
            # Determine environment
            env_type = detect_environment()
            
            # Initialize storage based on environment
            from config.storage import StorageFactory
            storage = StorageFactory.create(self.config)
            stratified_storage = storage['stratified_sample']
            
            # Load stratified data using appropriate storage
            stratified_data = await stratified_storage.get_sample()
            if stratified_data is None or stratified_data.empty:
                logger.warning("Loaded stratified data is empty")
                return pd.DataFrame()
            
            # Check if we need to add embeddings
            if 'embedding' not in stratified_data.columns:
                logger.info("Embeddings not present in stratified data, loading from Object Storage")
                
                # Get embeddings storage
                embedding_storage = storage['embeddings']
                
                # Get embeddings and thread map from storage
                embeddings_array, thread_id_map = await embedding_storage.get_embeddings()
                
                if embeddings_array is not None and thread_id_map is not None:
                    logger.info(f"Merging {len(embeddings_array)} embeddings with stratified data")
                    
                    # Add embedding column
                    stratified_data["embedding"] = None
                    
                    # Convert thread IDs to strings for consistent comparison
                    stratified_data["thread_id"] = stratified_data["thread_id"].astype(str)
                    thread_id_map = {str(k): v for k, v in thread_id_map.items()}
                    
                    # Map embeddings to rows
                    matched = 0
                    for idx, row in stratified_data.iterrows():
                        thread_id = str(row["thread_id"])
                        if thread_id in thread_id_map:
                            emb_idx = thread_id_map[thread_id]
                            if isinstance(emb_idx, (int, str)) and str(emb_idx).isdigit():
                                emb_idx = int(emb_idx)
                                if 0 <= emb_idx < len(embeddings_array):
                                    stratified_data.at[idx, "embedding"] = embeddings_array[emb_idx]
                                    matched += 1
                    
                    logger.info(f"Successfully matched {matched} embeddings out of {len(stratified_data)} rows")
                else:
                    logger.warning("Failed to load embeddings from storage")
            
            # Ensure we have the necessary text field for inference
            if "text_clean" not in stratified_data.columns and "content" in stratified_data.columns:
                logger.info("Adding text_clean field from content field")
                stratified_data["text_clean"] = stratified_data["content"]
            
            return stratified_data
            
        except Exception as e:
            logger.error(f"Error loading stratified data: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    async def needs_stratification(self, force_refresh: bool = False) -> bool:
        """Check if stratified sample needs to be created or updated.
        
        Args:
            force_refresh: Whether to force refresh regardless of current state
            
        Returns:
            bool: True if update is needed, False otherwise
        """
        # Check environment type
        env_type = detect_environment()
        
        if env_type.lower() == 'replit':
            # For Replit, use storage implementation
            from config.storage import StorageFactory
            storage = StorageFactory.create(self.config, 'replit')
            stratified_storage = storage['stratified_sample']
            
            try:
                # Check if stratified sample exists
                sample_exists = await stratified_storage.sample_exists()
                if not sample_exists:
                    self._logger.info("Stratified sample doesn't exist in Replit storage")
                    return True
                
                # If forcing refresh, always update
                if force_refresh:
                    self._logger.info("Force refresh requested, stratified sample will be updated")
                    return True
                
                self._logger.info("Stratified sample exists in Replit storage")
                return False
            except Exception as e:
                self._logger.error(f"Error checking stratified sample in Replit storage: {e}")
                return True
        else:
            # For file-based storage
            stratified_file = self.config.stratified_data_path / 'stratified_sample.csv'
            
            # Check if file exists
            if not stratified_file.exists():
                self._logger.info(f"Stratified sample file not found at {stratified_file}")
                return True
                
            # If forcing refresh, always update
            if force_refresh:
                self._logger.info("Force refresh requested, stratified sample will be updated")
                return True
                
            # Try to read the file to check if it has data
            try:
                sample_df = pd.read_csv(stratified_file)
                if sample_df.empty:
                    self._logger.info("Stratified sample is empty")
                    return True
                self._logger.info(f"Stratified sample exists with {len(sample_df)} rows")
                return False
            except Exception as e:
                self._logger.exception(f"Error checking stratified sample: {e}")
                return True

    async def is_data_ready(self, skip_embeddings: bool = False) -> bool:
        """
        Check if all necessary data is ready for inference.
        
        Args:
            skip_embeddings: If True, skips checking for embeddings
            
        Returns:
            bool: True if data is ready, False otherwise
        """
        logger.info(f"Checking if data is ready (skip_embeddings={skip_embeddings})")
        
        # Determine environment
        env_type = detect_environment()
        
        try:
            if env_type.lower() == 'replit':
                logger.info("Using Replit storage to check data readiness")
                # Initialize storage implementations for Replit
                from config.storage import StorageFactory
                storage = StorageFactory.create(self.config, 'replit')
                
                complete_data_storage = storage['complete_data']
                stratified_storage = storage['stratified_sample']
                embedding_storage = storage['embeddings']
                
                # Check if complete data exists
                row_count = await complete_data_storage.get_row_count()
                if row_count == 0:
                    logger.info("Complete data storage is empty, data not ready")
                    return False
                
                # Check if stratified sample exists
                stratified_exists = await stratified_storage.sample_exists()
                if not stratified_exists:
                    logger.info("Stratified sample does not exist, data not ready")
                    return False
                
                # Check if embeddings exist (unless skipped)
                if not skip_embeddings:
                    embeddings_exist = await embedding_storage.embeddings_exist()
                    if not embeddings_exist:
                        logger.info("Embeddings do not exist, data not ready")
                        return False
                
                logger.info("All required data exists in Replit storage, data is ready")
                return True
                
            else:
                # Local/Docker environment - use file-based approach
                logger.info("Using file-based storage to check data readiness")
                
                # Check if complete data exists
                complete_data_path = self.config.root_data_path / 'complete_data.csv'
                if not complete_data_path.exists():
                    logger.info("Complete data file doesn't exist, data not ready")
                    return False
                
                # Check if stratified data exists
                stratified_file = self.config.stratified_data_path / 'stratified_sample.csv'
                if not stratified_file.exists():
                    logger.info("Stratified data file doesn't exist, data not ready")
                    return False
                
                # Check if embeddings exist (unless skipped)
                if not skip_embeddings:
                    embeddings_path = self.config.stratified_data_path / 'embeddings.npz'
                    thread_id_map_path = self.config.stratified_data_path / 'thread_id_map.json'
                    
                    if not embeddings_path.exists() or not thread_id_map_path.exists():
                        logger.info("Embeddings or thread map file doesn't exist, data not ready")
                        return False
                    
                    # Verify files are valid
                    try:
                        # Try to load embedding files
                        np.load(embeddings_path)
                        with open(thread_id_map_path, 'r') as f:
                            thread_map = json.load(f)
                        
                        # Check if thread map is valid
                        if not isinstance(thread_map, dict) or len(thread_map) == 0:
                            logger.info("Thread map is invalid, data not ready")
                            return False
                    except Exception as e:
                        logger.warning(f"Error validating embedding files: {e}")
                        return False
                
                logger.info("All required data files exist, data is ready")
                return True
                
        except Exception as e:
            logger.error(f"Error checking if data is ready: {e}")
            logger.error(traceback.format_exc())
            return False
            
    async def load_stratified_data(self) -> pd.DataFrame:
        """
        Load stratified data for the knowledge agent.
        
        Returns:
            pd.DataFrame: Stratified data with embeddings column
        """
        logger.info("Using compatibility method load_stratified_data")
        try:
            # Call the internal method to load the stratified data
            data = await self._load_stratified_data()
            logger.info(f"Successfully loaded stratified data with {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error in load_stratified_data: {e}")
            logger.error(traceback.format_exc())
            
            # Fall back to a simple mock dataset if loading fails
            logger.warning("Falling back to mock data due to load error")
            return self._create_mock_data()
            
    def _create_mock_data(self) -> pd.DataFrame:
        """
        Create mock data for testing when real data is not available.
        
        Returns:
            pd.DataFrame: Mock data with basic fields
        """
        logger.info("Creating mock data for testing")
        mock_data = []
        num_records = 100
        
        for i in range(num_records):
            mock_data.append({
                'thread_id': str(10000000 + i),
                'created_at': datetime.now(pytz.UTC).isoformat(),
                'text_clean': f'Mock text for testing article {i}',
                'content': f'Mock content for article {i}',
                'embedding': np.random.rand(1536).astype(np.float32).tolist()  # Mock embedding vector
            })
        
        df = pd.DataFrame(mock_data)
        logger.info(f"Created mock data with {len(df)} records")
        return df