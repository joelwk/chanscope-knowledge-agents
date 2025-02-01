import pandas as pd
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from .data_processing.sampler import Sampler
from .data_processing.cloud_handler import load_all_csv_data_from_s3
from .data_processing.dialog_processor import process_references
from config.settings import Config
from knowledge_agents.data_processing.sampler import Sampler
import numpy as np
import gc

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

@dataclass
class DataConfig:
    """Configuration for data operations with validation."""
    root_path: Path
    all_data_path: Path
    stratified_data_path: Path
    knowledge_base_path: Path

    filter_date: str
    sample_size: int = field(default_factory=lambda: Config.SAMPLE_SIZE)
    time_column: str = 'posted_date_time'
    strata_column: Optional[str] = None
    
    # Chunk size and memory configurations
    processing_chunk_size: int = field(init=False)
    stratification_chunk_size: int = field(init=False)
    
    # Constants for optimization
    MAX_SAMPLE_SIZE: int = 100000  # Cap total sample size
    MIN_CHUNK_SIZE: int = 1000     # Minimum chunk size for statistical validity
    MAX_PROCESSING_CHUNK: int = 5000
    STRATIFICATION_RATIO: float = 0.5
    
    # DataFrame optimization configs
    dtype_optimizations: Dict[str, str] = field(default_factory=lambda: {
        'thread_id': 'str',
        'posted_date_time': 'str',
        'text_clean': 'str',
        'posted_comment': 'str'
    })
    
    def __post_init__(self):
        """Initialize and validate configurations."""
        # Validate and cap sample size
        if self.sample_size > self.MAX_SAMPLE_SIZE:
            logger.warning(f"Sample size exceeds limit of {self.MAX_SAMPLE_SIZE}. Capping sample size.")
            self.sample_size = self.MAX_SAMPLE_SIZE
            
        # Calculate optimal chunk sizes based on sample size
        self.processing_chunk_size = min(
            self.MAX_PROCESSING_CHUNK,
            max(self.MIN_CHUNK_SIZE, self.sample_size // 10)
        )
        
        # Stratification chunk size is proportional but never larger than processing
        self.stratification_chunk_size = min(
            int(self.processing_chunk_size * self.STRATIFICATION_RATIO),
            self.processing_chunk_size
        )
        
        # Ensure minimum chunk sizes
        self.stratification_chunk_size = max(self.MIN_CHUNK_SIZE, self.stratification_chunk_size)
        
        # Validate chunk size relationships
        if self.stratification_chunk_size > self.processing_chunk_size:
            raise ValueError("Stratification chunk size cannot exceed processing chunk size")

    @property
    def read_csv_kwargs(self) -> Dict[str, Any]:
        """Get optimized pandas read_csv parameters."""
        return {
            'dtype': self.dtype_optimizations,
            'usecols': list(self.dtype_optimizations.keys()),
            'on_bad_lines': 'warn',
            'low_memory': True}

    @classmethod
    def from_env(cls) -> 'DataConfig':
        """Create configuration from environment variables with validation."""
        root_path = Path(Config.ROOT_PATH)
        return cls(
            root_path=root_path,
            all_data_path=Path(Config.ALL_DATA),
            stratified_data_path=Path(Config.ALL_DATA_STRATIFIED_PATH),
            knowledge_base_path=Path(Config.KNOWLEDGE_BASE),
            sample_size=Config.SAMPLE_SIZE,
            filter_date=Config.FILTER_DATE if hasattr(Config, 'FILTER_DATE') else None,
            time_column=Config.TIME_COLUMN,
            strata_column=Config.STRATA_COLUMN)

class DataStateManager:
    """Manages data state and validation."""
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self.state_file = self.config.root_path / '.data_state'

    def _load_state(self) -> Dict[str, Any]:
        """Load data state from file."""
        if self.state_file.exists():
            try:
                return pd.read_json(self.state_file).to_dict(orient='records')[0]
            except Exception as e:
                self._logger.error(f"Failed to load state: {e}")
        return {'last_update': None, 'total_records': 0}

    def _save_state(self, state: Dict[str, Any]):
        """Save data state to file."""
        try:
            pd.DataFrame([state]).to_json(self.state_file)
        except Exception as e:
            self._logger.error(f"Failed to save state: {e}")

    def get_last_update(self) -> Optional[str]:
        """Get the last update timestamp."""
        return self._load_state().get('last_update')

    def update_state(self, total_records: int):
        """Update the data state."""
        state = {
            'last_update': pd.Timestamp.now(tz='UTC').isoformat(),
            'total_records': total_records
        }
        self._save_state(state)

    def _validate_path(self, path: Path, path_type: str) -> bool:
        """Validate a single path."""
        try:
            if not isinstance(path, (str, Path)):
                self._logger.error(f"Invalid {path_type} path type: {path}")
                return False
            return Path(path).exists()
        except Exception as e:
            self._logger.error(f"Path validation failed for {path_type}: {e}")
            return False

    def validate_file_structure(self) -> Dict[str, bool]:
        """Validate existence of required files and directories."""
        paths = {
            'root_dir': self.config.root_path,
            'all_data': self.config.all_data_path,
            'stratified_dir': self.config.stratified_data_path,
            'knowledge_base': self.config.knowledge_base_path
        }
        return {name: self._validate_path(path, name) for name, path in paths.items()}

    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate content of data files."""
        integrity = {'all_data': False, 'data_not_empty': False}
        
        # Validate sample size configuration
        if not isinstance(self.config.sample_size, int) or self.config.sample_size <= 0:
            self._logger.error(f"Invalid sample_size: {self.config.sample_size}")
            return integrity

        if not self.config.all_data_path.exists():
            return integrity

        try:
            # Read data in chunks to validate
            required_columns = [self.config.time_column]
            if self.config.strata_column:
                required_columns.append(self.config.strata_column)

            valid_rows = 0
            for chunk in pd.read_csv(self.config.all_data_path, 
                                   chunksize=self.config.processing_chunk_size,
                                   on_bad_lines='warn',
                                   low_memory=False):
                # Check required columns
                if all(col in chunk.columns for col in required_columns):
                    valid_rows += len(chunk)

                # Clean up
                del chunk
                gc.collect()

            integrity['all_data'] = True
            integrity['data_not_empty'] = valid_rows > 0
            self._logger.info(f"Data integrity check found {valid_rows} valid rows")

        except Exception as e:
            self._logger.error(f"Data integrity check failed: {e}")
        
        return integrity

    def check_data_integrity(self) -> bool:
        """Check if data files exist and are valid."""
        return all(self.validate_file_structure().values()) and all(self.validate_data_integrity().values())

class DataProcessor:
    """Handles data processing operations."""
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        # Use centralized chunk size configuration
        self.chunk_size = config.processing_chunk_size
        self.sampler = Sampler(
            filter_date=config.filter_date,
            time_column=config.time_column,
            strata_column=config.strata_column,
            initial_sample_size=config.sample_size
        )
        # Define required columns
        self.required_columns = {
            'thread_id': str,
            'posted_date_time': str,
            'text_clean': str,
            'posted_comment': str
        }

    async def stratify_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Stratify data with error handling and validation."""
        try:
            self._logger.info(f"Stratifying data with size {len(data)}")
            # Verify columns before stratification
            missing_cols = set(self.required_columns.keys()) - set(data.columns)
            if missing_cols:
                self._logger.error(f"Missing required columns: {missing_cols}")
                self._logger.info(f"Available columns: {data.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            stratified = self.sampler.stratified_sample(data)
            self._logger.info(f"Stratification complete. Result size: {len(stratified)}")
            return stratified
        except Exception as e:
            self._logger.error(f"Stratification failed: {e}")
            raise

class DataOperations:
    """Main data operations orchestrator."""
    def __init__(self, config: DataConfig):
        self.config = config
        self.state_manager = DataStateManager(config)
        self.processor = DataProcessor(config)
        self._logger = logging.getLogger(__name__)

    async def _fetch_and_save_data(self, force_refresh: bool = False):
        """Stream and process data ensuring complete temporal coverage with reservoir sampling."""
        self._logger.info(f"Fetching data with filter date: {self.config.filter_date}")
        try:
            # Get last update time if not forcing refresh
            last_update = None if force_refresh else self.state_manager.get_last_update()
            self._logger.info(f"Last update time: {last_update}")

            # Load existing knowledge base if doing incremental update
            existing_data = None
            if not force_refresh and self.config.knowledge_base_path.exists():
                try:
                    existing_data = pd.read_csv(self.config.knowledge_base_path)
                    existing_data[self.config.time_column] = pd.to_datetime(existing_data[self.config.time_column])
                    self._logger.info(f"Loaded existing knowledge base with {len(existing_data)} rows")
                except Exception as e:
                    self._logger.error(f"Failed to load existing knowledge base: {e}")
                    existing_data = None

            # Calculate target samples
            remaining_samples = self.config.sample_size - (len(existing_data) if existing_data is not None else 0)
            if remaining_samples <= 0:
                self._logger.info("Knowledge base already at target size")
                return

            self._logger.info(f"Target remaining samples: {remaining_samples}")

            # Initialize reservoir sampling by time period
            from collections import defaultdict
            import random
            reservoirs = defaultdict(list)
            period_counts = defaultdict(int)
            total_processed = 0
            
            # Single pass collection with reservoir sampling by time period
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=last_update):
                try:
                    # Create a copy to avoid SettingWithCopyWarning
                    chunk_df = chunk_df.copy()
                    
                    # Apply optimizations and conversions
                    for col, dtype in self.config.dtype_optimizations.items():
                        if col in chunk_df.columns:
                            chunk_df[col] = chunk_df[col].astype(dtype)
                    
                    # Convert time column and group by period
                    chunk_df[self.config.time_column] = pd.to_datetime(chunk_df[self.config.time_column])
                    
                    # Group by day period for stratified sampling
                    for period, period_data in chunk_df.groupby(chunk_df[self.config.time_column].dt.to_period('D')):
                        period_counts[period] += len(period_data)
                        
                        # Calculate target size for this period based on proportion of data seen
                        total_seen = sum(period_counts.values())
                        target_size = max(
                            int((remaining_samples * period_counts[period]) / total_seen),
                            min(100, remaining_samples // 10)  # Ensure minimum representation
                        )
                        
                        # Perform reservoir sampling for this period
                        current_samples = reservoirs[period]
                        for _, row in period_data.iterrows():
                            if len(current_samples) < target_size:
                                current_samples.append(row)
                            else:
                                # Randomly replace existing samples with decreasing probability
                                j = random.randint(0, period_counts[period] - 1)
                                if j < target_size:
                                    current_samples[j] = row
                    
                    total_processed += len(chunk_df)
                    if total_processed % 10000 == 0:
                        current_samples = sum(len(samples) for samples in reservoirs.values())
                        self._logger.info(f"Processed {total_processed} rows, current samples: {current_samples}")
                        
                except Exception as e:
                    self._logger.error(f"Error processing chunk: {e}")
                    continue

            # Combine samples from all periods
            if reservoirs:
                self._logger.info("Combining stratified samples from all periods...")
                period_samples = []
                total_samples = sum(len(samples) for samples in reservoirs.values())
                
                for period, samples in reservoirs.items():
                    if samples:
                        # Calculate final size for this period proportionally
                        period_df = pd.DataFrame(samples)
                        target_size = min(
                            int((remaining_samples * len(samples)) / total_samples),
                            remaining_samples - sum(len(df) for df in period_samples)
                        )
                        if len(period_df) > target_size:
                            period_df = period_df.sample(n=target_size)
                        period_samples.append(period_df)
                        self._logger.info(f"Period {period}: {len(period_df)} samples")

                final_data = pd.concat(period_samples, ignore_index=True)
                
                # Combine with existing data if doing incremental update
                if existing_data is not None and not force_refresh:
                    final_data = pd.concat([existing_data, final_data], ignore_index=True)

                # Final size check
                if len(final_data) > self.config.sample_size:
                    self._logger.info(f"Reducing final sample size from {len(final_data)} to {self.config.sample_size}")
                    final_data = final_data.sample(n=self.config.sample_size, random_state=42)

                # Save stratified sample
                stratified_file = self.config.stratified_data_path / "stratified_sample.csv"
                self._logger.info(f"Saving stratified sample to: {stratified_file.absolute()}")
                final_data.to_csv(stratified_file, index=False)
                self._logger.info(f"Saved stratified sample: {len(final_data)} rows")

                # Create/update knowledge base
                self._logger.info(f"Saving knowledge base to: {self.config.knowledge_base_path.absolute()}")
                final_data.to_csv(self.config.knowledge_base_path, index=False)
                self._logger.info(f"Created/updated knowledge base: {len(final_data)} rows")

                # Update state with new timestamp
                self.state_manager.update_state(len(final_data))

                # Clean up
                del final_data, period_samples, reservoirs
                gc.collect()

        except Exception as e:
            self._logger.error(f"Data fetch and processing failed: {e}")
            raise

    def _cleanup_existing_data(self):
        """Clean up existing data files while preserving other directories and files."""
        self._logger.info("Cleaning up data files")
        try:
            # Clean up state file
            if self.state_manager.state_file.exists():
                self.state_manager.state_file.unlink()
                self._logger.info(f"Removed state file: {self.state_manager.state_file}")

            # Clean up knowledge base file if it exists
            if self.config.knowledge_base_path.exists():
                self.config.knowledge_base_path.unlink()
                self._logger.info(f"Removed file: {self.config.knowledge_base_path}")

            # Clean up stratified data directory contents
            if self.config.stratified_data_path.exists():
                for item in self.config.stratified_data_path.glob("*"):
                    if item.is_file():
                        item.unlink()
                        self._logger.info(f"Removed file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self._logger.info(f"Removed directory: {item}")

            self._logger.info("Data cleanup complete")
        except Exception as e:
            self._logger.error(f"Cleanup failed: {e}")
            raise

    async def prepare_data(self, force_refresh: bool = False) -> str:
        """Main data preparation pipeline."""
        try:
            self._logger.info(f"Starting data preparation (force_refresh={force_refresh})")

            # Always perform cleanup if force refresh is True
            if force_refresh:
                self._cleanup_existing_data()

            # Create directory structure if needed
            await self._ensure_directory_structure()

            # Check if we need to load/refresh data
            structure_valid = self.state_manager.validate_file_structure()
            integrity_valid = self.state_manager.validate_data_integrity()
            
            if force_refresh or not all(structure_valid.values()) or not all(integrity_valid.values()):
                # Initial setup needed
                await self._fetch_and_save_data(force_refresh=True)
                return "Initial data preparation completed successfully"
            
            # Check if we have new data to process
            last_update = self.state_manager.get_last_update()
            if last_update:
                # Do incremental update
                await self._fetch_and_save_data(force_refresh=False)
                return "Incremental update completed successfully"
            else:
                # No state found, do full refresh
                await self._fetch_and_save_data(force_refresh=True)
                return "Full refresh completed due to missing state"

        except Exception as e:
            self._logger.error(f"Data preparation failed: {e}")
            raise

    async def _ensure_directory_structure(self):
        """Create necessary directories after cleanup."""
        self._logger.info("Creating directory structure")
        try:
            self.config.root_path.mkdir(parents=True, exist_ok=True)
            self.config.stratified_data_path.mkdir(parents=True, exist_ok=True)
            self.config.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            self._logger.info("Directory structure created successfully")
        except Exception as e:
            self._logger.error(f"Failed to create directory structure: {e}")
            raise

async def prepare_knowledge_base(force_refresh: bool = False) -> str:
    """Main entry point for data preparation."""
    try:
        config = DataConfig.from_env()
        operations = DataOperations(config)
        
        # First run the main data preparation
        result = await operations.prepare_data(force_refresh)
        logger.info(f"Data preparation result: {result}")
        
        # Then process references
        logger.info("Starting reference processing...")
        
        # Use the stratified data directory where stratified_sample.csv is located
        data_dir = str(config.stratified_data_path)
        stratified_file = Path(data_dir) / "stratified_sample.csv"
        
        # Verify the file exists and has content
        if not stratified_file.exists():
            raise FileNotFoundError(f"Stratified sample file not found at: {stratified_file.absolute()}")
            
        file_size = stratified_file.stat().st_size
        logger.info(f"Found stratified sample file: {stratified_file.absolute()} (size: {file_size} bytes)")
        
        if file_size == 0:
            raise ValueError(f"Stratified sample file is empty: {stratified_file.absolute()}")
        
        # Only process references if force_refresh or if knowledge base doesn't exist
        if force_refresh or not config.knowledge_base_path.exists():
            logger.info(f"Processing references from stratified directory: {data_dir}")
            output_path = await process_references(output_dir=data_dir)
            logger.info(f"Reference processing completed. Output saved to: {output_path}")
            return "Knowledge base preparation completed successfully"
        else:
            logger.info("Using existing knowledge base, skipping reference processing")
            return "Using existing knowledge base"
        
    except Exception as e:
        logger.error(f"Knowledge base preparation failed: {e}")
        raise