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
            'low_memory': True
        }

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

    async def _fetch_and_save_data(self):
        """Stream and process data ensuring complete temporal coverage."""
        self._logger.info(f"Fetching data with filter date: {self.config.filter_date}")
        try:
            # Track periods and their sizes for proportional sampling
            period_stats = {}
            total_rows = 0
            
            # First pass: Collect period statistics
            self._logger.info("First pass: Collecting temporal distribution statistics")
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=self.config.filter_date):
                try:
                    # Safe type conversion with error handling
                    try:
                        chunk_df[self.config.time_column] = pd.to_datetime(chunk_df[self.config.time_column])
                    except Exception as e:
                        self._logger.error(f"Failed to convert time column: {e}")
                        continue

                    # Update period statistics
                    periods = chunk_df[self.config.time_column].dt.to_period('D')
                    period_counts = periods.value_counts()
                    
                    for period, count in period_counts.items():
                        if period not in period_stats:
                            period_stats[period] = 0
                        period_stats[period] += count
                        total_rows += count
                    
                finally:
                    del chunk_df
            
            if not period_stats:
                raise ValueError("No valid data found in the specified date range")
            
            self._logger.info(f"Found {len(period_stats)} time periods with {total_rows} total rows")
            
            # Calculate target samples per period
            samples_per_period = {
                period: max(
                    int(self.config.sample_size * count / total_rows),
                    self.config.MIN_CHUNK_SIZE
                )
                for period, count in period_stats.items()
            }
            
            # Second pass: Collect stratified samples
            self._logger.info("Second pass: Collecting stratified samples")
            stratified_samples = []
            current_period_samples = {}
            
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=self.config.filter_date):
                try:
                    # Apply optimizations and conversions
                    for col, dtype in self.config.dtype_optimizations.items():
                        if col in chunk_df.columns:
                            try:
                                chunk_df[col] = chunk_df[col].astype(dtype)
                            except Exception as e:
                                self._logger.warning(f"Failed to convert column {col} to {dtype}: {e}")
                    
                    chunk_df[self.config.time_column] = pd.to_datetime(chunk_df[self.config.time_column])
                    
                    # Process each period in the chunk
                    for period, period_data in chunk_df.groupby(chunk_df[self.config.time_column].dt.to_period('D')):
                        if period not in samples_per_period:
                            continue
                            
                        target_size = samples_per_period[period]
                        current_size = len(current_period_samples.get(period, []))
                        
                        if current_size < target_size:
                            # Stratify this batch
                            stratified_batch = await self.processor.stratify_data(period_data)
                            if not stratified_batch.empty:
                                if period not in current_period_samples:
                                    current_period_samples[period] = []
                                current_period_samples[period].append(stratified_batch)
                                
                                # If we have enough samples for this period, finalize it
                                total_period_samples = sum(len(df) for df in current_period_samples[period])
                                if total_period_samples >= target_size:
                                    combined_period = pd.concat(current_period_samples[period], ignore_index=True)
                                    if len(combined_period) > target_size:
                                        combined_period = combined_period.sample(n=target_size)
                                    stratified_samples.append(combined_period)
                                    del current_period_samples[period]
                                    self._logger.info(f"Completed sampling for period {period}: {len(combined_period)} rows")
                
                finally:
                    del chunk_df
            
            # Finalize any remaining periods
            for period, period_samples in current_period_samples.items():
                if period_samples:
                    combined_period = pd.concat(period_samples, ignore_index=True)
                    target_size = samples_per_period[period]
                    if len(combined_period) > target_size:
                        combined_period = combined_period.sample(n=target_size)
                    stratified_samples.append(combined_period)
                    self._logger.info(f"Finalized remaining period {period}: {len(combined_period)} rows")
            
            # Combine and save final stratified sample
            if stratified_samples:
                final_data = pd.concat(stratified_samples, ignore_index=True)
                
                # Apply maximum sample size limit
                MAX_TOTAL_SAMPLES = 5000  # Limit total samples to keep embedding process manageable
                if len(final_data) > MAX_TOTAL_SAMPLES:
                    self._logger.info(f"Reducing final sample size from {len(final_data)} to {MAX_TOTAL_SAMPLES}")
                    final_data = final_data.sample(n=MAX_TOTAL_SAMPLES, random_state=42)
                
                # Verify DataFrame contents before saving
                self._logger.info(f"Final data columns: {final_data.columns.tolist()}")
                self._logger.info(f"Final data shape: {final_data.shape}")
                
                # Save stratified sample
                stratified_file = self.config.stratified_data_path / "stratified_sample.csv"
                self._logger.info(f"Saving stratified sample to: {stratified_file.absolute()}")
                final_data.to_csv(stratified_file, index=False)
                self._logger.info(f"Saved stratified sample: {len(final_data)} rows")
                
                # Create knowledge base
                self._logger.info(f"Saving knowledge base to: {self.config.knowledge_base_path.absolute()}")
                final_data.to_csv(self.config.knowledge_base_path, index=False)
                self._logger.info(f"Created knowledge base: {len(final_data)} rows")
                
                # Clean up
                del final_data
            
            # Clean up
            del stratified_samples
            del current_period_samples
            
        except Exception as e:
            self._logger.error(f"Data fetch and processing failed: {e}")
            raise

    async def prepare_data(self, force_refresh: bool = False) -> str:
        """Main data preparation pipeline."""
        try:
            self._logger.info(f"Starting data preparation (force_refresh={force_refresh})")

            # Always perform cleanup if force refresh is True
            if force_refresh:
                self._cleanup_existing_data()

            # Create directory structure first
            await self._ensure_directory_structure()

            # Check if we need to load/refresh data
            structure_valid = self.state_manager.validate_file_structure()
            if force_refresh or not all(structure_valid.values()):
                await self._fetch_and_save_data()
                return "Data preparation completed successfully"

            # Validate data integrity if not forcing refresh
            integrity_valid = self.state_manager.validate_data_integrity()
            if not all(integrity_valid.values()):
                raise ValueError("Data integrity check failed")

            self._logger.info("Using existing valid data")
            return "Using existing valid data"

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

    def _cleanup_existing_data(self):
        """Clean up existing data files while preserving other directories and files."""
        self._logger.info("Cleaning up data files")
        try:
            # Only clean up specific data files and directories
            if self.config.all_data_path.exists():
                self.config.all_data_path.unlink()
                self._logger.info(f"Removed file: {self.config.all_data_path}")

            # Clean up stratified data directory contents
            if self.config.stratified_data_path.exists():
                for item in self.config.stratified_data_path.glob("*"):
                    if item.is_file():
                        item.unlink()
                        self._logger.info(f"Removed file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self._logger.info(f"Removed directory: {item}")
                self._logger.info(f"Cleaned stratified data directory: {self.config.stratified_data_path}")

            # Clean up knowledge base file if it exists
            if self.config.knowledge_base_path.exists():
                self.config.knowledge_base_path.unlink()
                self._logger.info(f"Removed file: {self.config.knowledge_base_path}")

            self._logger.info("Data cleanup complete")
        except Exception as e:
            self._logger.error(f"Cleanup failed: {e}")
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
        
        logger.info(f"Processing references from stratified directory: {data_dir}")
        output_path = await process_references(output_dir=data_dir)
        logger.info(f"Reference processing completed. Output saved to: {output_path}")
        
        return "Knowledge base preparation completed successfully"
        
    except Exception as e:
        logger.error(f"Knowledge base preparation failed: {e}")
        raise