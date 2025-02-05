import pandas as pd
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from .data_processing.sampler import Sampler
from .data_processing.cloud_handler import load_all_csv_data_from_s3
from .data_processing.dialog_processor import process_references
from config.settings import Config
from knowledge_agents.data_processing.sampler import Sampler
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
    """Configuration class for data operations."""
    root_data_path: Path
    stratified_data_path: Path
    knowledge_base_path: Path
    temp_path: Path
    filter_date: Optional[str] = None
    sample_size: int = 1000
    time_column: str = 'posted_date_time'
    strata_column: str = 'thread_id'
    
    def __post_init__(self):
        """Initialize and validate configurations."""
        # Get centralized settings
        chunk_settings = Config.get_chunk_settings()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()
        
        # Initialize from settings
        self.dtype_optimizations = column_settings['column_types']
        
        # Set chunk sizes from centralized settings
        self.processing_chunk_size = chunk_settings['processing_chunk_size']
        self.stratification_chunk_size = chunk_settings['stratification_chunk_size']
        
        # Set sample size with validation
        self.sample_size = sample_settings['default_sample_size']
        if self.sample_size > sample_settings['max_sample_size']:
            logger.warning(f"Sample size exceeds limit of {sample_settings['max_sample_size']}. Capping sample size.")
            self.sample_size = sample_settings['max_sample_size']
        elif self.sample_size < sample_settings['min_sample_size']:
            logger.warning(f"Sample size below minimum of {sample_settings['min_sample_size']}. Setting to minimum.")
            self.sample_size = sample_settings['min_sample_size']
        
        # Ensure all paths are Path objects
        self.root_data_path = Path(self.root_data_path)
        self.stratified_data_path = Path(self.stratified_data_path)
        self.knowledge_base_path = Path(self.knowledge_base_path)
        self.temp_path = Path(self.temp_path)

    @property
    def read_csv_kwargs(self) -> Dict[str, Any]:
        """Get optimized pandas read_csv parameters."""
        return {
            'dtype': self.dtype_optimizations,
            'on_bad_lines': 'warn',
            'low_memory': True
        }

    @classmethod
    def from_config(cls) -> 'DataConfig':
        """Create DataConfig instance from Config settings."""
        paths = Config.get_paths()
        column_settings = Config.get_column_settings()
        sample_settings = Config.get_sample_settings()
        
        # Create base data directory if it doesn't exist
        data_path = Path(paths['root_data_path']) / 'data'
        data_path.mkdir(parents=True, exist_ok=True)
        
        return cls(
            root_data_path=Path(paths['root_data_path']),
            stratified_data_path=Path(paths['stratified']),
            knowledge_base_path=Path(paths['knowledge_base']),
            temp_path=Path(paths['temp']),
            filter_date=Config.get_filter_date(),
            sample_size=sample_settings['default_sample_size'],
            time_column=column_settings['time_column'],
            strata_column=column_settings['strata_column']
        )

class DataStateManager:
    """Manages data state and validation."""
    def __init__(self, config: DataConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self.state_file = self.config.root_data_path / '.data_state'

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
        # Map config paths to their validation requirements
        paths = {
            'root_data_path': self.config.root_data_path,
            'stratified': self.config.stratified_data_path,
            'knowledge_base': self.config.knowledge_base_path,
            'temp': self.config.temp_path
        }
        
        # Ensure directories exist for file paths
        for path_key in ['knowledge_base']:
            if path_key in paths and paths[path_key].parent:
                paths[path_key].parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure directory paths exist
        for path_key in ['root_data_path', 'stratified', 'temp']:
            if path_key in paths:
                paths[path_key].mkdir(parents=True, exist_ok=True)
        
        # Validate all paths
        validation_results = {}
        for name, path in paths.items():
            try:
                # For directories, check if they exist and are directories
                if name in ['root_data_path', 'stratified', 'temp']:
                    validation_results[name] = path.exists() and path.is_dir()
                # For files, just check if parent directory exists
                else:
                    validation_results[name] = path.parent.exists()
                
                if not validation_results[name]:
                    self._logger.error(f"Path validation failed for {name}: {path} does not exist")
            except Exception as e:
                self._logger.error(f"Path validation failed for {name}: {str(e)}")
                validation_results[name] = False
        
        return validation_results

    def validate_data_integrity(self) -> Dict[str, bool]:
        """Validate content of data files."""
        integrity = {'data_not_empty': False}
        
        # Validate sample size configuration
        if not isinstance(self.config.sample_size, int) or self.config.sample_size <= 0:
            self._logger.error(f"Invalid sample_size: {self.config.sample_size}")
            return integrity

        if not self.config.root_data_path.exists():
            return integrity

        try:
            # Check if knowledge base exists and has valid content
            if self.config.knowledge_base_path.exists():
                try:
                    df = pd.read_csv(self.config.knowledge_base_path)
                    integrity['data_not_empty'] = len(df) > 0
                    self._logger.info(f"Knowledge base validation: {len(df)} rows found")
                except Exception as e:
                    self._logger.error(f"Failed to validate knowledge base: {e}")
            
            return integrity

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
        logger.info(f"DataOperations.__init__ received filter_date: {config.filter_date}")
        self.state_manager = DataStateManager(config)
        self.processor = DataProcessor(config)
        self._logger = logging.getLogger(__name__)

    async def _fetch_and_save_data(self, force_refresh: bool = False):
        """Stream and process data ensuring complete temporal coverage with reservoir sampling."""
        self._logger.info("=== Starting Data Fetch and Save ===")
        self._logger.info(f"Force refresh: {force_refresh}")
        self._logger.info(f"Config filter_date: {self.config.filter_date}")
        
        try:
            # Get last update time if not forcing refresh
            last_update = self.config.filter_date if force_refresh else self.state_manager.get_last_update()
            self._logger.info(f"Last update time: {last_update}")
            self._logger.info(f"Using filter date: {self.config.filter_date}")

            # Load existing knowledge base if doing incremental update
            existing_data = None
            if not force_refresh and self.config.knowledge_base_path.exists():
                try:
                    existing_data = pd.read_csv(self.config.knowledge_base_path)
                    existing_data[self.config.time_column] = pd.to_datetime(existing_data[self.config.time_column])
                    self._logger.info(f"Loaded existing knowledge base with {len(existing_data)} rows")
                    
                    # If we have enough samples and not forcing refresh, return early
                    if len(existing_data) >= self.config.sample_size:
                        self._logger.info("Existing knowledge base already at target size and force_refresh=False")
                        return
                        
                except Exception as e:
                    self._logger.error(f"Failed to load existing knowledge base: {e}")
                    existing_data = None

            # Calculate target samples
            remaining_samples = self.config.sample_size - (len(existing_data) if existing_data is not None else 0)
            if remaining_samples <= 0:
                self._logger.info("Knowledge base already at target size")
                return

            self._logger.info(f"Target remaining samples: {remaining_samples}")
            
            # Check if we need to process new data
            if not force_refresh and existing_data is not None and len(existing_data) > 0:
                self._logger.info("Checking for new data since last update...")
                last_processed_date = existing_data[self.config.time_column].max()
                self._logger.info(f"Last processed date: {last_processed_date}")
                
                if last_processed_date >= pd.Timestamp(self.config.filter_date):
                    self._logger.info("No new data to process - existing data is up to date")
                    return
                else:
                    self._logger.info(f"Processing new data from {last_processed_date} to {self.config.filter_date}")
                    last_update = last_processed_date

            # Initialize reservoir sampling by time period
            from collections import defaultdict
            import random
            reservoirs = defaultdict(list)
            period_counts = defaultdict(int)
            total_processed = 0
            
            # Log initial state
            self._logger.info("=== Starting Data Loading ===")
            self._logger.info(f"Filter date: {self.config.filter_date}")
            self._logger.info(f"Last update: {last_update}")
            
            # Single pass collection with reservoir sampling by time period
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=last_update):
                try:
                    # Log chunk info
                    self._logger.info(f"=== Processing New Chunk ===")
                    self._logger.info(f"Chunk size before processing: {len(chunk_df)}")
                    self._logger.info(f"Chunk columns: {chunk_df.columns.tolist()}")
                    
                    # Create a deep copy and reset index to avoid RangeIndex issues
                    chunk_df = chunk_df.copy(deep=True)
                    chunk_df.index = pd.RangeIndex(len(chunk_df))  # Explicitly set a new RangeIndex
                    chunk_df.reset_index(drop=True, inplace=True)
                    
                    # Convert columns using dtype_optimizations from config
                    for col, dtype in self.config.dtype_optimizations.items():
                        if col in chunk_df.columns:
                            try:
                                self._logger.info(f"Converting column '{col}' to dtype '{dtype}'")
                                chunk_df[col] = chunk_df[col].astype(dtype)
                            except (TypeError, ValueError) as e:
                                self._logger.warning(
                                    f"Failed to convert column '{col}' to dtype '{dtype}': {str(e)}. "
                                    "Skipping conversion for this column. Please review column type settings in configuration."
                                )
                                continue
                            except Exception as e:
                                self._logger.error(f"Unexpected error converting column '{col}' to dtype '{dtype}': {str(e)}")
                                continue
                    
                    # Convert time column to UTC and group by period
                    try:
                        # Log the state of the DataFrame before time conversion
                        self._logger.info(f"Processing chunk with {len(chunk_df)} rows")
                        self._logger.debug(f"Chunk columns before time conversion: {chunk_df.columns.tolist()}")
                        
                        # Convert time column to UTC
                        chunk_df[self.config.time_column] = pd.to_datetime(
                            chunk_df[self.config.time_column], 
                            utc=True,
                            errors='coerce'
                        )
                        
                        # Log after time conversion
                        self._logger.info(f"Rows after time conversion: {len(chunk_df)}")
                        if len(chunk_df) > 0:
                            self._logger.info(f"Time range in chunk: {chunk_df[self.config.time_column].min()} to {chunk_df[self.config.time_column].max()}")
                        
                        # Filter out invalid dates and reset index
                        chunk_df = chunk_df.dropna(subset=[self.config.time_column])
                        chunk_df = chunk_df.reset_index(drop=True)
                        
                        # Log after filtering
                        self._logger.info(f"Rows after filtering invalid dates: {len(chunk_df)}")
                        
                        # Create a copy before groupby to avoid RangeIndex issues
                        chunk_df = chunk_df.copy()
                        
                        # Convert time column to period before groupby
                        period_series = chunk_df[self.config.time_column].dt.tz_localize(None).dt.to_period('D')
                        
                        # Log period information
                        unique_periods = period_series.unique()
                        self._logger.info(f"Unique periods in chunk: {len(unique_periods)}")
                        if len(unique_periods) > 0:
                            self._logger.info(f"Period range: {unique_periods[0]} to {unique_periods[-1]}")
                        
                        # Group by date period
                        for period, group_indices in period_series.groupby(period_series).groups.items():
                            # Get period data using integer location to avoid index issues
                            period_data = chunk_df.iloc[group_indices].copy()
                            period_data.reset_index(drop=True, inplace=True)
                            
                            period_counts[period] += len(period_data)
                            
                            # Log period processing
                            self._logger.info(f"Processing period {period}: {len(period_data)} rows")
                            
                            # Calculate target size for this period based on proportion of data seen
                            total_seen = sum(period_counts.values())
                            target_size = max(
                                int((remaining_samples * period_counts[period]) / total_seen),
                                min(100, remaining_samples // 10)  # Ensure minimum representation
                            )
                            
                            # Log sampling information
                            self._logger.info(f"Period {period} - Target size: {target_size}, Total seen: {total_seen}")
                            
                            # Perform reservoir sampling for this period
                            current_samples = reservoirs[period]
                            for idx, row in period_data.iterrows():
                                if len(current_samples) < target_size:
                                    current_samples.append(row.to_dict())
                                else:
                                    j = random.randint(0, period_counts[period] - 1)
                                    if j < target_size:
                                        current_samples[j] = row.to_dict()
                            
                            # Log reservoir state
                            self._logger.info(f"Period {period} - Current samples: {len(current_samples)}")
                        
                        total_processed += len(chunk_df)
                        if total_processed % 10000 == 0:
                            current_samples = sum(len(samples) for samples in reservoirs.values())
                            self._logger.info(f"Processed {total_processed} rows, current samples: {current_samples}")
                            
                    except Exception as e:
                        self._logger.error(f"Error processing time data in chunk: {str(e)}")
                        self._logger.error(f"Chunk info - shape: {chunk_df.shape}, dtypes: {chunk_df.dtypes}")
                        continue
                        
                except Exception as e:
                    self._logger.error(f"Error processing chunk: {str(e)}")
                    continue

            # Combine samples from all periods
            if reservoirs:
                self._logger.info(f"Combining stratified samples from {len(reservoirs)} periods...")
                self._logger.info(f"Total samples before combining: {sum(len(samples) for samples in reservoirs.values())}")
                period_samples = []
                total_samples = sum(len(samples) for samples in reservoirs.values())
                
                if total_samples == 0:
                    self._logger.error("No samples collected during processing")
                    raise ValueError("No samples collected during processing")
                
                for period, samples in reservoirs.items():
                    if samples:
                        try:
                            # Convert list of dicts to DataFrame
                            period_df = pd.DataFrame.from_records(samples)
                            # Calculate final size for this period proportionally
                            target_size = min(
                                int((remaining_samples * len(samples)) / total_samples),
                                remaining_samples - sum(len(df) for df in period_samples)
                            )
                            if len(period_df) > target_size:
                                period_df = period_df.sample(n=target_size)
                            period_samples.append(period_df)
                            self._logger.info(f"Period {period}: {len(period_df)} samples")
                        except Exception as e:
                            self._logger.error(f"Error processing period {period}: {str(e)}")
                            continue

                if not period_samples:
                    self._logger.error("No valid period samples after processing")
                    raise ValueError("No valid period samples after processing")

                final_data = pd.concat(period_samples, ignore_index=True)
                self._logger.info(f"Combined data shape: {final_data.shape}")
                
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

            # Create directory structure first
            await self._ensure_directory_structure()

            # Now validate the structure after we've created the directories
            structure_valid = self.state_manager.validate_file_structure()
            if not all(structure_valid.values()):
                missing = [k for k, v in structure_valid.items() if not v]
                self._logger.error(f"Missing required paths in configuration: {', '.join(missing)}")
                raise ValueError(f"Missing required paths in configuration: {', '.join(missing)}")

            # Check data integrity
            integrity_valid = self.state_manager.validate_data_integrity()
            
            if force_refresh or not all(integrity_valid.values()):
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
            # Create all required directories
            self.config.root_data_path.mkdir(parents=True, exist_ok=True)
            self.config.stratified_data_path.mkdir(parents=True, exist_ok=True)
            self.config.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.temp_path.mkdir(parents=True, exist_ok=True)
            
            self._logger.info("Directory structure created successfully")
        except Exception as e:
            self._logger.error(f"Failed to create directory structure: {e}")
            raise

async def prepare_knowledge_base(force_refresh: bool = False, config: Optional[DataConfig] = None) -> str:
    """Main entry point for data preparation."""
    try:
        # Use provided config or create new one
        if config is None:
            config = DataConfig.from_config()
            logger.info("Created new DataConfig from config")
        logger.info(f"prepare_knowledge_base using filter_date: {config.filter_date}")
        
        operations = DataOperations(config)
        logger.info(f"DataOperations initialized with filter_date: {operations.config.filter_date}")
        
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