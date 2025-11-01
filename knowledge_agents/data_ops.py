import asyncio
import json
import logging
import os
import pytz
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Union, Callable, Awaitable
from filelock import FileLock, Timeout

import numpy as np
import pandas as pd

from config.base_settings import get_base_settings
from config.settings import Config
from .data_processing.sampler import Sampler
from config.env_loader import detect_environment
from knowledge_agents.utils import validate_text

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
    strata_column: Optional[str] = None
    board_id: Optional[str] = None
    force_refresh: bool = False
    env: Optional[str] = None

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
        
        # Get current environment
        env = detect_environment()

        return cls(
            root_data_path=root_data_path,
            stratified_data_path=stratified_path,
            temp_path=temp_path,
            filter_date=filter_date,
            sample_size=sample_size,
            time_column=time_column,
            strata_column=strata_column,
            force_refresh=force_refresh,
            env=env
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
            'posted_date_time': str,
            'text_clean': str,
        }

    async def stratify_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self._logger.info(f"Stratifying data with {len(data)} records")
        missing = set(self.required_columns.keys()) - set(data.columns)
        if missing:
            self._logger.error(f"Missing columns: {missing}. Available: {data.columns.tolist()}")
            raise ValueError(f"Missing columns: {missing}")
            
        # Validate text quality before stratification
        if 'text_clean' in data.columns:
            text_valid_rows = []
            invalid_rows = 0
            invalid_reasons = {}
            
            for idx, row in data.iterrows():
                is_valid, reason = validate_text(row['text_clean'])
                if is_valid:
                    text_valid_rows.append(idx)
                else:
                    invalid_rows += 1
                    if reason not in invalid_reasons:
                        invalid_reasons[reason] = 0
                    invalid_reasons[reason] += 1
            
            if invalid_rows > 0:
                self._logger.info(f"Filtered out {invalid_rows} rows with invalid text before stratification")
                for reason, count in invalid_reasons.items():
                    self._logger.debug(f"  - {reason}: {count} rows")
                
                # Keep only rows with valid text
                data = data.loc[text_valid_rows]
                self._logger.info(f"Proceeding with stratification using {len(data)} valid records")
                
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

        # Initialize storage implementations
        from config.storage import StorageFactory
        storage = StorageFactory.create(self.config)
        
        # Initialize embedding generator from embedding_ops
        from knowledge_agents.embedding_ops import generate_embeddings
        
        # Initialize storages
        self.embedding_storage = storage['embeddings']
        self.complete_data_storage = storage['complete_data']
        self.stratified_storage = storage['stratified_sample']

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
                    last_update = last_update.replace(tzinfo=pytz.UTC)
                return (datetime.now(pytz.UTC) - last_update).total_seconds() < 3600  # 1 hour
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
            # Import inference_ops to clear caches when embeddings are updated
            from knowledge_agents.inference_ops import _clear_caches
            # Initialize embedding storage using environment-aware factory
            embedding_storage = self.embedding_storage
            if embedding_storage is None:
                from config.storage import StorageFactory
                embedding_storage = StorageFactory.create(self.config).get('embeddings')
                if embedding_storage is None:
                    logger.error("Failed to initialize embedding storage")
                    return

            logger.info(f"Starting embedding update (force_refresh={force_refresh})")
            
            # Check incremental embeddings setting
            base_settings = get_base_settings()
            incremental_embeddings = base_settings.get('processing', {}).get('incremental_embeddings', True)
            
            # Check if we need to update embeddings
            if not force_refresh and incremental_embeddings:
                logger.info("Checking for incremental embedding updates")
                embeddings_exist = await embedding_storage.embeddings_exist()
                if embeddings_exist:
                    # Load existing embeddings and thread map
                    existing_embeddings, existing_thread_map = await embedding_storage.get_embeddings()
                    if existing_embeddings is not None and existing_thread_map is not None:
                        # Load current stratified data
                        stratified_data = await self._load_stratified_data()
                        if stratified_data is None or len(stratified_data) == 0:
                            logger.error("No stratified data available for embedding generation")
                            return
                        
                        # Check for new articles
                        existing_thread_ids = set(existing_thread_map.keys())
                        current_thread_ids = set(stratified_data['thread_id'].astype(str).tolist())
                        new_thread_ids = current_thread_ids - existing_thread_ids
                        
                        if not new_thread_ids:
                            logger.info("No new articles found, embeddings are up to date")
                            return
                        
                        logger.info(f"Found {len(new_thread_ids)} new articles for incremental embedding update")
                        
                        # Filter new articles
                        new_articles_data = stratified_data[stratified_data['thread_id'].astype(str).isin(new_thread_ids)]
                        
                        # Generate embeddings for new articles only
                        new_content = new_articles_data['text_clean'].tolist()
                        new_thread_ids_list = new_articles_data['thread_id'].astype(str).tolist()
                        
                        # Process in batches
                        batch_size = 100
                        new_embeddings = []
                        
                        from knowledge_agents.embedding_ops import generate_embeddings
                        
                        for i in range(0, len(new_content), batch_size):
                            batch_content = new_content[i:i+batch_size]
                            batch_thread_ids = new_thread_ids_list[i:i+batch_size]
                            
                            batch_num = i//batch_size + 1
                            total_batches = (len(new_content) + batch_size - 1) // batch_size
                            logger.info(f"Processing incremental embedding batch {batch_num}/{total_batches}")
                            
                            # Update progress if callback provided
                            if progress_callback:
                                progress_percent = int((i / len(new_content)) * 100)
                                if asyncio.iscoroutinefunction(progress_callback):
                                    await progress_callback(progress_percent, 100)
                                else:
                                    progress_callback(progress_percent, 100)
                            
                            try:
                                batch_embeddings = await generate_embeddings(batch_content)
                                if batch_embeddings:
                                    new_embeddings.extend(batch_embeddings)
                                    logger.info(f"Generated {len(batch_embeddings)} new embeddings")
                            except Exception as e:
                                logger.error(f"Error generating embeddings for batch {batch_num}: {e}")
                                continue
                        
                        if new_embeddings:
                            # Combine existing and new embeddings
                            combined_embeddings = np.vstack([existing_embeddings, np.array(new_embeddings, dtype=np.float32)])
                            
                            # Update thread ID map
                            updated_thread_map = existing_thread_map.copy()
                            base_index = len(existing_embeddings)
                            for i, thread_id in enumerate(new_thread_ids_list[:len(new_embeddings)]):
                                updated_thread_map[thread_id] = base_index + i
                            
                            # Store updated embeddings
                            success = await embedding_storage.store_embeddings(combined_embeddings, updated_thread_map)
                            
                            if success:
                                logger.info(f"Successfully updated embeddings with {len(new_embeddings)} new entries (total: {len(combined_embeddings)})")
                                # Clear caches after embedding update
                                _clear_caches()
                                return
                            else:
                                logger.error("Failed to store incremental embedding updates, falling back to full refresh")
                                force_refresh = True
                        else:
                            logger.warning("No new embeddings generated, falling back to full refresh")
                            force_refresh = True
                    else:
                        logger.info("Could not load existing embeddings, performing full refresh")
                        force_refresh = True
                else:
                    logger.info("No existing embeddings found, performing full generation")
            elif force_refresh:
                logger.info("Force refresh enabled, will regenerate all embeddings")
            else:
                logger.info("Incremental embeddings disabled, checking if full generation is needed")
                embeddings_exist = await embedding_storage.embeddings_exist()
                if embeddings_exist:
                    logger.info("Embeddings already exist and force_refresh is False, skipping update")
                    return
            
            # Full embedding generation (original logic)
            logger.info("Performing full embedding generation")
            
            # Load stratified data
            logger.info("Loading stratified data for embedding generation")
            stratified_data = await self._load_stratified_data()
            if stratified_data is None or len(stratified_data) == 0:
                logger.error("No stratified data available for embedding generation")
                return
            
            logger.info(f"Loaded stratified data with {len(stratified_data)} rows")
            
            # Get thread IDs and content for embedding generation
            if 'thread_id' not in stratified_data.columns:
                logger.error("Required 'thread_id' column not found in stratified data")
                return
            
            if 'text_clean' not in stratified_data.columns:
                logger.error("Required 'text_clean' column not found in stratified data")
                return
            
            thread_ids = stratified_data['thread_id'].astype(str).tolist()
            content = stratified_data['text_clean'].tolist()
            
            if not thread_ids or not content:
                logger.error("No content or thread IDs available for embedding generation")
                return
            
            logger.info(f"Preparing to generate embeddings for {len(thread_ids)} items")
            
            # Process in batches to avoid memory issues
            batch_size = 100
            all_embeddings = []
            thread_id_map = {}
            
            # Import the generate_embeddings function
            from knowledge_agents.embedding_ops import generate_embeddings
            
            total_batches = (len(content) + batch_size - 1) // batch_size
            logger.info(f"Processing embeddings in {total_batches} batches of up to {batch_size} items each")
            
            for i in range(0, len(content), batch_size):
                batch_text = content[i:i+batch_size]
                batch_ids = thread_ids[i:i+batch_size]
                
                batch_num = i//batch_size + 1
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch_text)} items)")
                
                # Update progress if callback provided
                if progress_callback:
                    progress_percent = int((i / len(content)) * 100)
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(progress_percent, 100)
                    else:
                        progress_callback(progress_percent, 100)
                
                # Generate embeddings for batch
                try:
                    batch_embeddings = await generate_embeddings(batch_text)
                    
                    if batch_embeddings is None:
                        logger.error(f"Failed to generate embeddings for batch {batch_num}")
                        continue
                    
                    # Add to results
                    for j, embedding in enumerate(batch_embeddings):
                        idx = len(all_embeddings)
                        all_embeddings.append(embedding)
                        thread_id_map[batch_ids[j]] = idx
                    
                    logger.info(f"Completed batch {batch_num}/{total_batches} with {len(batch_embeddings)} embeddings")
                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_num}: {batch_error}")
                    logger.error(traceback.format_exc())
                    continue
                
                await asyncio.sleep(0.1)  # Small delay to prevent API rate limits
            
            # Update final progress
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(100, 100)
                else:
                    progress_callback(100, 100)
            
            if not all_embeddings:
                logger.error("No embeddings were generated, cannot continue with storage")
                return
            
            # Convert to numpy array for storage
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            logger.info(f"Generated {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")
            
            # Store embeddings in Object Storage
            logger.info(f"Storing embeddings and thread ID map in Object Storage")
            success = await embedding_storage.store_embeddings(embeddings_array, thread_id_map)
            
            if success:
                logger.info(f"Successfully generated and stored {len(all_embeddings)} embeddings in Object Storage")
                
                # Clear caches after embedding update
                _clear_caches()
                
                # Verify embeddings were stored correctly
                logger.info("Verifying embeddings were stored correctly")
                try:
                    verification_embeddings, verification_thread_map = await embedding_storage.get_embeddings()
                    if verification_embeddings is not None and verification_thread_map is not None:
                        logger.info(f"Verification successful: Retrieved {len(verification_embeddings)} embeddings")
                    else:
                        logger.warning("Verification failed: Could not retrieve stored embeddings")
                except Exception as verify_error:
                    logger.warning(f"Verification check failed with error: {verify_error}")
            else:
                logger.error("Failed to store embeddings in Object Storage")
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {e}")
            logger.error(traceback.format_exc())

    async def ensure_data_ready(
        self,
        force_refresh: bool = False,
        skip_embeddings: bool = False,
        max_workers: Optional[int] = None
    ) -> bool:
        """Ensure all required data is ready for processing.
        
        Args:
            force_refresh: Whether to force refresh of all data
            skip_embeddings: Whether to skip embedding generation
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            bool: True if data is ready, False otherwise
        """
        try:
            # Import inference_ops to clear caches when data is refreshed
            from knowledge_agents.inference_ops import _clear_caches
            
            logger.info(f"Ensuring data is ready (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})")
            
            # Check environment type to determine appropriate data preparation methods
            from config.env_loader import detect_environment
            env_type = detect_environment()
            
            if env_type.lower() == 'replit':
                logger.info("Using Replit-specific data preparation")
                # Use Replit-specific storage implementations
                from config.storage import StorageFactory
                storage = StorageFactory.create(self.config, env_type)
                
                # Check if complete data exists in PostgreSQL
                complete_data_storage = storage['complete_data']
                row_count = await complete_data_storage.get_row_count()
                logger.info(f"PostgreSQL database has {row_count} rows")
                
                # If no data or force refresh, fetch from S3
                if row_count == 0 or force_refresh:
                    logger.info("Fetching data from S3 and storing in PostgreSQL")
                    success = await self._fetch_and_store_s3_data()
                    if not success:
                        logger.error("Failed to fetch and store S3 data")
                        return False
                    
                    # Clear caches after data refresh
                        _clear_caches()
                
                # Check if stratified sample exists
                stratified_storage = storage['stratified_sample']
                stratified_exists = await stratified_storage.sample_exists()
                
                if not stratified_exists or force_refresh:
                    logger.info("Creating stratified sample")
                    success = await self._create_stratified_sample()
                    if not success:
                        logger.error("Failed to create stratified sample")
                        return False
                    
                    # Clear caches after stratification
                    _clear_caches()
                
                # Generate embeddings if not skipping
                if not skip_embeddings:
                        await self._update_embeddings(force_refresh=force_refresh, max_workers=max_workers)

            else:
                # Use standard file-based approach for Docker/local environment
                logger.info("Using file-based data preparation")
                
                # Check if complete data exists
                complete_data_path = self.config.root_data_path / 'complete_data.csv'
                if not complete_data_path.exists() or force_refresh:
                    logger.info("Fetching and processing S3 data")
                    success = await self._fetch_and_store_s3_data()
                    if not success:
                        logger.error("Failed to fetch and store S3 data")
                        return False
                    
                    # Clear caches after data refresh
                    _clear_caches()
                
                # Check if stratified sample exists
                stratified_path = self.config.stratified_data_path / 'stratified_sample.csv'
                if not stratified_path.exists() or force_refresh:
                    logger.info("Creating stratified sample")
                    success = await self._create_stratified_sample()
                    if not success:
                        logger.error("Failed to create stratified sample")
                        return False
                    
                    # Clear caches after stratification
                    _clear_caches()
                
                # Generate embeddings if not skipping
                if not skip_embeddings:
                    embeddings_path = self.config.stratified_data_path / 'embeddings.npz'
                    thread_map_path = self.config.stratified_data_path / 'thread_id_map.json'
                    
                    if not embeddings_path.exists() or not thread_map_path.exists() or force_refresh:
                        await self._update_embeddings(force_refresh=force_refresh, max_workers=max_workers)

            logger.info("Data preparation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error ensuring data ready: {e}")
            logger.error(traceback.format_exc())
            return False

    async def _fetch_and_store_s3_data(self) -> bool:
        """Fetch data from S3 and store it using the active storage backend.

        - In Replit (PostgreSQL-backed), delegate to storage.prepare_data().
        - In file-based environments, download and write CSV locally.
        """
        try:
            # Prefer a storage-provided implementation if available (Replit path)
            if hasattr(self.complete_data_storage, "prepare_data"):
                logger.info("Delegating S3 fetch to storage.prepare_data()")
                return await self.complete_data_storage.prepare_data()

            # Fallback to local file-based implementation
            logger.info("Using internal downloader for S3 -> local CSV")
            return await self._download_and_process_data()
        except Exception as e:
            logger.error(f"Error in _fetch_and_store_s3_data: {e}")
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

    async def _create_stratified_sample(self) -> bool:
        """Create and store a stratified sample from the complete dataset.

        Returns True on successful storage, False otherwise.
        """
        try:
            self._logger.info("Creating stratified sample from complete data")
            
            # Load complete data via the active storage backend
            complete_df = await self.complete_data_storage.get_data(
                filter_date=self.config.filter_date
            )
            if complete_df is None or len(complete_df) == 0:
                self._logger.error("Complete dataset is empty or unavailable; cannot stratify")
                return False

            # Ensure we have a usable text column
            if 'text_clean' not in complete_df.columns:
                if 'content' in complete_df.columns:
                    self._logger.warning("text_clean not found; falling back to content column for stratification")
                    complete_df['text_clean'] = complete_df['content']
                else:
                    self._logger.error("Neither text_clean nor content column present; cannot stratify")
                    return False

            # Perform stratification
            stratified_df = await self.processor.stratify_data(complete_df)
            
            # Persist using the active storage
            storage_success = await self.stratified_storage.store_sample(stratified_df)
            if storage_success:
                self._logger.info(f"Stored stratified sample with {len(stratified_df)} rows")
                return True
            
            self._logger.error("Failed to store stratified sample")
            return False
        except Exception as e:
            self._logger.error(f"Error creating stratified sample: {e}")
            self._logger.error(traceback.format_exc())
            return False

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
            
            # Import embedding generation function
            from knowledge_agents.embedding_ops import generate_embeddings
            
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
                batch_embeddings = await generate_embeddings(batch_text)
                
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
            storage = StorageFactory.create(self.config, env_type)
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
            # Load stratified data using appropriate storage
            stratified_data = await self.stratified_storage.get_sample()
            if stratified_data is None or stratified_data.empty:
                logger.warning("Loaded stratified data is empty")
                return pd.DataFrame()
            
            # Check if we need to add embeddings
            if 'embedding' not in stratified_data.columns:
                logger.info("Embeddings not present in stratified data, loading from Object Storage")
                
                try:
                    # Try to get embeddings from Object Storage
                    embeddings_array, thread_id_map = await self.embedding_storage.get_embeddings()
                    
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
                        logger.warning("Failed to load embeddings from Object Storage, generating new embeddings")
                        # Generate new embeddings for the data
                        from knowledge_agents.embedding_ops import generate_embeddings
                        text_content = stratified_data['text_clean'].tolist()
                        embeddings = await generate_embeddings(text_content)
                        if embeddings is not None:
                            stratified_data['embedding'] = embeddings
                            logger.info(f"Generated {len(embeddings)} new embeddings")
                            
                except Exception as e:
                    logger.warning(f"Error loading embeddings from Object Storage: {e}")
                    logger.info("Generating new embeddings for the data")
                    # Generate new embeddings for the data
                    from knowledge_agents.embedding_ops import generate_embeddings
                    text_content = stratified_data['text_clean'].tolist()
                    embeddings = await generate_embeddings(text_content)
                    if embeddings is not None:
                        stratified_data['embedding'] = embeddings
                        logger.info(f"Generated {len(embeddings)} new embeddings")
            
            # Ensure we have the necessary text field for inference
            if "text_clean" not in stratified_data.columns and "content" in stratified_data.columns:
                logger.warning("text_clean column not found in stratified data, mapping from content column as fallback")
                stratified_data["text_clean"] = stratified_data["content"]
                logger.warning("This mapping from content to text_clean should be temporary. The primary flow should store text_clean in the database.")
            
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
            storage = StorageFactory.create(self.config, env_type)
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
                storage = StorageFactory.create(self.config, env_type)
                
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
