import pandas as pd
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from .data_processing.sampler import Sampler
from .data_processing.cloud_handler import load_all_csv_data_from_s3
from .stratified_ops import split_dataframe
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
# Prevent propagation to root logger to avoid duplicate logs
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

    def validate_file_structure(self) -> Dict[str, bool]:
        """Validate existence of required files and directories."""
        return {
            'root_dir': self.config.root_path.exists(),
            'all_data': self.config.all_data_path.exists(),
            'stratified_dir': self.config.stratified_data_path.exists(),
            'knowledge_base': self.config.knowledge_base_path.exists()
        }

    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate content of data files."""
        integrity = {}
        if self.config.all_data_path.exists():
            try:
                # Read data in chunks to validate
                required_columns = [self.config.time_column]
                if self.config.strata_column:
                    required_columns.append(self.config.strata_column)
                
                valid_rows = 0
                for chunk in pd.read_csv(self.config.all_data_path, 
                                       chunksize=Config.SAMPLE_SIZE,
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
                integrity['all_data'] = False
                integrity['data_not_empty'] = False
        return integrity

    def check_data_integrity(self) -> bool:
        """Check if data files exist and are valid."""
        try:
            # Validate configuration
            if not isinstance(self.config.sample_size, int) or self.config.sample_size <= 0:
                logger.error(f"Invalid sample_size: {self.config.sample_size}")
                return False

            # Check if required paths exist
            required_paths = [
                self.config.root_path,
                self.config.all_data_path,
                self.config.stratified_data_path,
                self.config.knowledge_base_path
            ]

            for path in required_paths:
                if not isinstance(path, (str, Path)):
                    logger.error(f"Invalid path type for {path}")
                    return False
                if not Path(path).exists():
                    logger.error(f"Required path does not exist: {path}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Data integrity check failed: {str(e)}")
            return False

class DataProcessor:
    """Handles data processing operations."""
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self.sampler = Sampler(
            filter_date=config.filter_date,
            time_column=config.time_column,
            strata_column=config.strata_column,
            initial_sample_size=config.sample_size
        )

    async def stratify_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Stratify data with error handling and validation."""
        try:
            self._logger.info(f"Stratifying data with size {len(data)}")
            stratified = self.sampler.stratified_sample(data)
            self._logger.info(f"Stratification complete. Result size: {len(stratified)}")
            return stratified
        except Exception as e:
            self._logger.error(f"Stratification failed: {e}")
            raise

    async def split_data(self, data: pd.DataFrame) -> None:
        """Split data into train/test sets."""
        try:
            self._logger.info("Splitting stratified data")
            # Ensure required columns are present
            required_columns = {'thread_id', 'posted_date_time', 'text_clean'}
            missing_cols = required_columns - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns for stratification: {missing_cols}")
            # Only keep necessary columns before splitting
            data_to_split = data[list(required_columns)]
            # Split the data
            split_dataframe(
                data_to_split,
                fraction=0.1,
                stratify_column=self.config.time_column,
                save_directory=str(self.config.stratified_data_path),
                seed=42,
                file_format='csv'
            )
            self._logger.info("Data splitting complete")
        except Exception as e:
            self._logger.error(f"Data splitting failed: {e}")
            raise

class DataOperations:
    """Main data operations orchestrator."""
    def __init__(self, config: DataConfig):
        self.config = config
        self.state_manager = DataStateManager(config)
        self.processor = DataProcessor(config)
        self._logger = logging.getLogger(__name__)

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
                await self._load_fresh_data()
            
            # Validate data integrity
            integrity_valid = self.state_manager.validate_data_integrity()
            if not all(integrity_valid.values()):
                raise ValueError("Data integrity check failed")
            
            # Process existing data
            await self._process_existing_data()
            
            self._logger.info("Data preparation completed successfully")
            return "Data preparation completed successfully"
            
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

    async def _load_fresh_data(self):
        """Load fresh data from source."""
        self._logger.info("Loading fresh data from source")
        # Ensure directory structure exists before fetching data
        await self._ensure_directory_structure()
        await self._fetch_and_save_data()

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
        
    async def _fetch_and_save_data(self):
        """Fetch and save new data in chunks."""
        self._logger.info(f"Fetching new data with filter date: {self.config.filter_date}")
        try:
            # Ensure parent directory exists before saving
            self.config.all_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize CSV file with headers
            pd.DataFrame(columns=['thread_id', 'posted_date_time', 'text_clean']).to_csv(
                self.config.all_data_path, index=False, mode='w')
            
            # Stream data directly from S3 to disk
            total_rows = 0
            
            # Iterate through chunks yielded by load_all_csv_data_from_s3
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=self.config.filter_date):
                # Append chunk directly to file
                chunk_df.to_csv(
                    self.config.all_data_path,
                    index=False,
                    mode='a',
                    header=False
                )
                total_rows += len(chunk_df)
                self._logger.info(f"Saved chunk with {len(chunk_df)} rows. Total rows: {total_rows}")
                
                # Clean up
                del chunk_df
                gc.collect()
            
            if total_rows == 0:
                self._logger.warning("No new data was saved")
            else:
                self._logger.info(f"New data saved successfully: {total_rows} rows")
            
        except Exception as e:
            self._logger.error(f"Data fetch failed: {e}")
            raise

    async def _process_existing_data(self):
        """Process and stratify existing data in chunks."""
        self._logger.info("Processing existing data")
        try:
            chunk_size = 50000  # Process 50k rows at a time
            all_data = []  # Store all chunks
            
            # Read and process data in chunks
            for chunk in pd.read_csv(self.config.all_data_path, chunksize=chunk_size):
                # Stratify chunk
                stratified_chunk = await self.processor.stratify_data(chunk)
                if not stratified_chunk.empty:
                    all_data.append(stratified_chunk)
                
                # Clean up
                del chunk
                gc.collect()
            
            # Combine all processed chunks
            if all_data:
                stratified_data = pd.concat(all_data, ignore_index=True)
                self._logger.info(f"Combined stratified data size: {len(stratified_data)}")
                
                # Save stratified sample with clear naming
                stratified_file = self.config.stratified_data_path / "stratified_sample.csv"
                stratified_data.to_csv(stratified_file, index=False)
                self._logger.info(f"Saved stratified sample to {stratified_file}")
                
                # Create knowledge base from stratified data
                self._logger.info("Creating knowledge base")
                stratified_data.to_csv(self.config.knowledge_base_path, index=False)
                self._logger.info(f"Knowledge base created with {len(stratified_data)} rows")
                
                # Clean up
                del stratified_data
                del all_data
                gc.collect()
                
            self._logger.info("Data processing complete")
            
        except Exception as e:
            self._logger.error(f"Data processing failed: {e}")
            raise

    def process_data(self, force_refresh: bool = False) -> bool:
        """Process data files and prepare for knowledge agent operations."""
        try:
            # Create required directories
            self._create_directory_structure()

            # Check if processing is needed
            if not force_refresh and self._is_processing_complete():
                logger.info("Data processing already complete")
                return True

            # Read and process data in chunks
            logger.info("Processing data files...")
            valid_rows = 0
            for chunk in pd.read_csv(self.config.all_data_path, 
                                  chunksize=self.config.sample_size,
                                  on_bad_lines='warn',
                                  low_memory=False):
                # Process chunk
                processed_chunk = self._process_chunk(chunk)
                if processed_chunk is not None:
                    valid_rows += len(processed_chunk)

            logger.info(f"Processed {valid_rows} valid rows")
            return True

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return False

async def prepare_knowledge_base(force_refresh: bool = False) -> str:
    """Main entry point for data preparation."""
    config = DataConfig.from_env()
    operations = DataOperations(config)
    return await operations.prepare_data(force_refresh)