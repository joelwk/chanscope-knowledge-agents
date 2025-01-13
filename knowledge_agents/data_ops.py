import pandas as pd
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from .data_processing.sampler import Sampler
from .data_processing.cloud_handler import load_all_csv_data_from_s3
from .stratified_ops import split_dataframe
from config.settings import Config
from knowledge_agents.data_processing.sampler import Sampler
import numpy as np
import gc

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass
class DataConfig:
    """Configuration for data operations with validation."""
    root_path: Path
    all_data_path: Path
    stratified_data_path: Path
    knowledge_base_path: Path
    sample_size: int
    filter_date: str
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
            sample_size=Config.DEFAULT_BATCH_SIZE,
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
                df = pd.read_csv(self.config.all_data_path)
                required_columns = [self.config.time_column]
                if self.config.strata_column:
                    required_columns.append(self.config.strata_column)
                integrity['all_data'] = all(col in df.columns for col in required_columns)
                integrity['data_not_empty'] = len(df) > 0
            except Exception as e:
                self._logger.error(f"Data integrity check failed: {e}")
                integrity['all_data'] = False
                integrity['data_not_empty'] = False
        return integrity

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
            split_dataframe(
                data,
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
            
            # Load data in chunks and save directly
            chunk_size = 50000  # Process 50k rows at a time
            new_data = load_all_csv_data_from_s3(latest_date_processed=self.config.filter_date)
            
            # Save in chunks to avoid memory issues
            if not new_data.empty:
                for i, chunk in enumerate(np.array_split(new_data, max(1, len(new_data) // chunk_size))):
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0
                    chunk_df = pd.DataFrame(chunk)
                    chunk_df.to_csv(self.config.all_data_path, index=False, mode=mode, header=header)
                    del chunk_df
                    gc.collect()
                
                self._logger.info(f"New data saved successfully: {len(new_data)} rows")
            else:
                self._logger.warning("No new data to save")
                
            # Clean up
            del new_data
            gc.collect()
            
        except Exception as e:
            self._logger.error(f"Data fetch failed: {e}")
            raise

    async def _process_existing_data(self):
        """Process and stratify existing data in chunks."""
        self._logger.info("Processing existing data")
        try:
            chunk_size = 50000  # Process 50k rows at a time
            processed_chunks = []
            
            # Read and process data in chunks
            for chunk in pd.read_csv(self.config.all_data_path, chunksize=chunk_size):
                # Stratify chunk
                stratified_chunk = await self.processor.stratify_data(chunk)
                processed_chunks.append(stratified_chunk)
                
                # Clean up
                del chunk
                gc.collect()
            
            # Combine processed chunks
            if processed_chunks:
                stratified_data = pd.concat(processed_chunks, ignore_index=True)
                
                # Split and save
                await self.processor.split_data(stratified_data)
                
                # Clean up
                del stratified_data
                del processed_chunks
                gc.collect()
                
            self._logger.info("Data processing complete")
            
        except Exception as e:
            self._logger.error(f"Data processing failed: {e}")
            raise

async def prepare_knowledge_base(force_refresh: bool = False) -> str:
    """Main entry point for data preparation."""
    config = DataConfig.from_env()
    operations = DataOperations(config)
    return await operations.prepare_data(force_refresh)