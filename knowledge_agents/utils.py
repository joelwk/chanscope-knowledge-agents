import os
import pandas as pd
import logging
import warnings
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple, List, Union
import traceback
import hashlib

# Import centralized logging configuration
from config.logging_config import get_logger

# Create a logger using the centralized configuration
logger = get_logger('knowledge_agents.utils')

warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_date(date_str, default_format="%Y-%m-%d %H:%M:%S"):
    """Parse a date string into a datetime object."""
    if isinstance(date_str, datetime):
        return date_str
    
    if not date_str:
        return None
        
    formats = [
        default_format,
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

def safe_str_to_date(date_str, format="%Y-%m-%d %H:%M:%S"):
    try:
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, format)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting '{date_str}' to date format '{format}': {e}")
        return None

def within_date_range(date, start_date, end_date):
    if date is None:
        return False
    return (start_date is None or date >= start_date) and (end_date is None or date <= end_date)

def count_total_rows(directory):
    total_rows = 0
    files_count = 0
    for file_name in [f for f in os.listdir(directory) if f.endswith('.parquet')]:
        total_rows += len(pd.read_parquet(os.path.join(directory, file_name)))
        files_count += 1
    print(f'Total number of files processed {files_count} containing: {total_rows} rows')
    return total_rows

class DateProcessor:
    def __init__(self, local_timezone='America/New_York'):
        self.local_timezone = local_timezone

    def _to_datetime(self, df, columns):
        for col in columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        return df

    def _localize_and_convert_to_utc(self, df, columns):
        for col in columns:
            df[col] = df[col].dt.tz_localize(self.local_timezone).dt.tz_convert(None)
        return df

    def _shift_time(self, df, column, hours=0):
        df[column] = df[column] - pd.Timedelta(hours=hours)
        return df

    def format_dates(self, df: pd.DataFrame, posted_col='posted_date_time', collected_col='collected_date_time') -> pd.DataFrame:
        """
        Process date columns to ensure correct timezone localization, conversion to UTC, and floor date to hour.
        """
        # Convert columns to datetime
        df = self._to_datetime(df, [posted_col, collected_col])

        # Drop rows with invalid dates
        df.dropna(subset=[posted_col, collected_col], inplace=True)

        # Adjust collected date by subtracting 1 hour
        df = self._shift_time(df, collected_col, hours=2)

        # Floor dates to the nearest hour
        df['collected_date_hour'] = df[collected_col].dt.floor('H')

        # Format dates as strings
        df[collected_col] = df[collected_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['collected_date_hour'] = df['collected_date_hour'].dt.strftime('%Y-%m-%d %H')
        return df

    def add_days(self, df: pd.DataFrame, date_column: str, days: int) -> pd.DataFrame:
        """
        Add a specified number of days to a datetime column.
        """
        df[date_column] = pd.to_datetime(df[date_column], utc=True, errors='coerce') + pd.Timedelta(days=days)
        return df

    def subtract_days(self, df: pd.DataFrame, date_column: str, days: int) -> pd.DataFrame:
        """
        Subtract a specified number of days from a datetime column.
        """
        df[date_column] = pd.to_datetime(df[date_column], utc=True, errors='coerce') - pd.Timedelta(days=days)
        return df

    def filter_date_range(self, df: pd.DataFrame, date_column: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter rows based on a date range.
        """
        df[date_column] = pd.to_datetime(df[date_column], utc=True, errors='coerce')
        start_date = pd.to_datetime(start_date, utc=True)
        end_date = pd.to_datetime(end_date, utc=True)
        return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
        
    def format_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date_hour' in df.columns:
            df['date_hour'] = pd.to_datetime(df['date_hour'], utc=True, errors='coerce')
            df['date_hour'] = df['date_hour'].dt.floor('H').dt.strftime('%Y-%m-%d %H')
        elif 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
            df['Timestamp'] = df['Timestamp'].dt.floor('H')
            df['date_hour'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H')
        else:
            raise ValueError("No 'date_hour' or 'Timestamp' column found in DataFrame.")
        return df
    
    def to_string(self, timestamp, format='%Y-%m-%d %H:%M:%S'):
        return timestamp.strftime(format) if pd.notnull(timestamp) else None

async def save_query_output(
    response: Optional[Dict[str, Any]] = None, 
    result: Optional[Dict[str, Any]] = None,
    base_path: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
    include_embeddings: bool = True,
    save_numpy: bool = True,
    query: Optional[str] = None,
    task_id: Optional[str] = None
) -> Tuple[Optional[Union[Path, str]], Optional[Union[Path, str]]]:
    """Save query output to JSON and optionally save embeddings to a separate file.
    
    Args:
        response: Full response to save (with or without embeddings)
        result: Legacy parameter (use response instead)
        base_path: Base directory to save output to
        logger: Optional logger instance
        include_embeddings: Whether to include embeddings in output
        save_numpy: Whether to save embeddings in NumPy format
        query: Original query that generated the response
        task_id: Task ID associated with the response
        
    Returns:
        Tuple of (path to JSON output, path to embeddings file if saved)
    """
    # Setup logger if not provided
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Support both 'response' and legacy 'result' parameter
    if response is None and result is not None:
        response = result
    
    if response is None:
        logger.warning("No response or result provided to save_query_output")
        return None, None
    
    # Detect environment
    from config.env_loader import is_replit_environment
    is_replit = is_replit_environment()
    
    # Set default base path if not provided (for file system fallback)
    if base_path is None:
        try:
            from config.settings import Config
            base_path = Path(Config.get_paths().get('generated_data', 'data/generated_data'))
        except Exception as e:
            logger.warning(f"Error getting base path from Config: {e}")
            base_path = Path('data/generated_data')
    
    # Convert string path to Path object if needed
    if isinstance(base_path, str):
        base_path = Path(base_path)
    
    # Generate filename details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if task_id:
        filename_prefix = f"query_{task_id}_{timestamp}"
    else:
        # Generate a short hash of the query if available
        if query:
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            filename_prefix = f"query_{query_hash}_{timestamp}"
        else:
            filename_prefix = f"query_{timestamp}"
    
    # Process response (create copies)
    response_without_embeddings = response.copy()
    embeddings_data = {}
    
    # Add the original query to the main response if provided
    if query and "query" not in response_without_embeddings:
        response_without_embeddings["query"] = query
    
    # Add task_id to the main response if provided
    if task_id and "task_id" not in response_without_embeddings:
        response_without_embeddings["task_id"] = task_id
    
    # If embeddings are requested, fetch them for each thread
    if include_embeddings:
        try:
            thread_ids = [chunk["thread_id"] for chunk in response.get("chunks", [])]
            if thread_ids:
                thread_embeddings = await _fetch_embeddings_for_threads(thread_ids)
                
                # Add embeddings to response if found
                if thread_embeddings:
                    for chunk in response.get("chunks", []):
                        thread_id = chunk.get("thread_id")
                        if thread_id and thread_id in thread_embeddings:
                            # Store embeddings in the separate data structure
                            embeddings_data[thread_id] = thread_embeddings[thread_id]
        except Exception as embedding_error:
            logger.warning(f"Error processing embeddings: {embedding_error}")
            # Continue without embeddings
    
    # Try to use Object Storage for Replit environment
    if is_replit:
        try:
            # Save to Object Storage
            result = await _save_to_object_storage(
                response_without_embeddings,
                embeddings_data,
                filename_prefix,
                include_embeddings,
                save_numpy,
                logger
            )
            
            if result[0] is not None:  # If successful, return the result
                logger.info("Successfully saved query output to Object Storage")
                return result
            
            # If unsuccessful, fall back to file system
            logger.warning("Object Storage save failed, falling back to file system")
        except Exception as e:
            logger.warning(f"Error using Object Storage, falling back to file system: {e}")
    
    # Use file system (either as primary or fallback)
    return await _save_to_filesystem(
        response_without_embeddings,
        embeddings_data,
        base_path,
        filename_prefix,
        include_embeddings,
        save_numpy,
        logger
    )

async def _save_to_object_storage(
    response_json: Dict[str, Any],
    embeddings_data: Dict[str, List[float]],
    filename_prefix: str,
    include_embeddings: bool,
    save_numpy: bool,
    logger: logging.Logger
) -> Tuple[Optional[str], Optional[str]]:
    """Save query output to Replit Object Storage.
    
    Args:
        response_json: Response JSON to save
        embeddings_data: Embeddings data to save
        filename_prefix: Prefix for the filename
        include_embeddings: Whether to include embeddings
        save_numpy: Whether to save embeddings in NumPy format
        logger: Logger instance
        
    Returns:
        Tuple of (path to JSON output, path to embeddings file if saved)
    """
    try:
        # Import Object Storage client
        from replit.object_storage import Client
        
        # Initialize client with proper error handling
        try:
            client = Client()
        except ValueError as bucket_error:
            # Handle no default bucket error
            if "no default bucket" in str(bucket_error).lower():
                logger.warning("No default bucket configured. Using 'knowledge-agent'")
                client = Client(bucket='query_results')
            else:
                raise
        
        # Save main JSON output
        json_key = f"query_results/{filename_prefix}.json"
        json_content = json.dumps(response_json, indent=2, ensure_ascii=False)
        client.upload_from_text(json_key, json_content)
        logger.info(f"Saved query output to Object Storage: {json_key}")
        
        # Save embeddings if available
        embeddings_key = None
        if embeddings_data and include_embeddings:
            embeddings_key = f"query_results/embeddings/{filename_prefix}_embeddings.json"
            embeddings_content = json.dumps(embeddings_data, indent=2, ensure_ascii=False)
            client.upload_from_text(embeddings_key, embeddings_content)
            logger.info(f"Saved embeddings to Object Storage: {embeddings_key}")
            
            # Save NumPy format if requested
            if save_numpy and embeddings_data:
                try:
                    # Convert to numpy array
                    import numpy as np
                    embeddings_list = list(embeddings_data.values())
                    if embeddings_list:
                        embeddings_array = np.array(embeddings_list, dtype=np.float32)
                        
                        # Save to temporary file first
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
                            np.savez_compressed(temp_file.name, embeddings=embeddings_array)
                            temp_path = temp_file.name
                        
                        # Upload from temporary file
                        numpy_key = f"query_results/embeddings/{filename_prefix}_embeddings.npz"
                        with open(temp_path, 'rb') as f:
                            numpy_data = f.read()
                        client.upload_from_bytes(numpy_key, numpy_data)
                        logger.info(f"Saved NumPy embeddings to Object Storage: {numpy_key}")
                        
                        # Clean up temp file
                        import os
                        os.remove(temp_path)
                except Exception as numpy_error:
                    logger.warning(f"Error saving NumPy embeddings to Object Storage: {numpy_error}")
        
        return json_key, embeddings_key
        
    except Exception as e:
        logger.error(f"Error saving to Object Storage: {e}")
        logger.error(traceback.format_exc())
        return None, None

async def _save_to_filesystem(
    response_json: Dict[str, Any],
    embeddings_data: Dict[str, List[float]],
    base_path: Path,
    filename_prefix: str,
    include_embeddings: bool,
    save_numpy: bool,
    logger: logging.Logger
) -> Tuple[Optional[Path], Optional[Path]]:
    """Save query output to the file system.
    
    Args:
        response_json: Response JSON to save
        embeddings_data: Embeddings data to save
        base_path: Base directory to save output to
        filename_prefix: Prefix for the filename
        include_embeddings: Whether to include embeddings
        save_numpy: Whether to save embeddings in NumPy format
        logger: Logger instance
        
    Returns:
        Tuple of (path to JSON output, path to embeddings file if saved)
    """
    try:
        # Create base directory if it doesn't exist
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create base directory {base_path}: {e}")
            # Try fallback to current directory
            base_path = Path('.')
        
        # Create embeddings directory
        embeddings_dir = base_path / "embeddings"
        try:
            embeddings_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create embeddings directory {embeddings_dir}: {e}")
            # Use base path as fallback
            embeddings_dir = base_path
        
        # Prepare paths
        json_path = base_path / f"{filename_prefix}.json"
        embeddings_json_path = embeddings_dir / f"{filename_prefix}_embeddings.json"
        embeddings_npy_path = embeddings_dir / f"{filename_prefix}_embeddings.npz"
        
        # Save main JSON output without embeddings
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(response_json, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved query output to {json_path}")
        except Exception as json_error:
            logger.error(f"Error saving main JSON output: {json_error}")
            return None, None
        
        # Save embeddings to separate files if we have any
        embeddings_path = None
        if embeddings_data and include_embeddings:
            try:
                # Save as JSON
                with open(embeddings_json_path, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
                embeddings_path = embeddings_json_path
                logger.info(f"Saved embeddings to {embeddings_json_path}")
                
                # Save as NumPy if requested
                if save_numpy:
                    try:
                        import numpy as np
                        embeddings_list = list(embeddings_data.values())
                        if embeddings_list:
                            embeddings_array = np.array(embeddings_list, dtype=np.float32)
                            np.savez_compressed(embeddings_npy_path, embeddings=embeddings_array)
                            logger.info(f"Also saved embeddings in NumPy format to {embeddings_npy_path}")
                    except Exception as numpy_error:
                        logger.warning(f"Error saving NumPy embeddings: {numpy_error}")
            except Exception as embedding_save_error:
                logger.warning(f"Error saving embeddings: {embedding_save_error}")
                # Continue with main JSON only
        
        return json_path, embeddings_path
        
    except Exception as e:
        logger.error(f"Error saving to filesystem: {e}")
        logger.error(traceback.format_exc())
        
        # Try simplified save as fallback
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                simplified_response = {
                    "summary": response_json.get("summary", ""),
                    "query": response_json.get("query", ""),
                    "task_id": response_json.get("task_id", ""),
                    "timestamp": datetime.now().isoformat(),
                    "error": "Error during full save, using simplified format"
                }
                json.dump(simplified_response, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved simplified output to {json_path} after error")
            return json_path, None
        except Exception as fallback_error:
            logger.error(f"Fallback save also failed: {fallback_error}")
            return None, None

async def _fetch_embeddings_for_threads(thread_ids: List[str]) -> Dict[str, List[float]]:
    """Fetch embeddings for a list of thread IDs from Object Storage.
    
    Args:
        thread_ids: List of thread IDs to fetch embeddings for
        
    Returns:
        Dictionary mapping thread IDs to their embeddings
    """
    try:
        # Import Object Storage client
        try:
            from replit.object_storage import Client
            from config.storage import ReplitObjectEmbeddingStorage
            from config.chanscope_config import ChanScopeConfig
        except ImportError as e:
            logging.warning(f"Could not import Object Storage modules: {e}")
            return {}
        
        # Initialize storage
        try:
            config = ChanScopeConfig.from_env()
            embedding_storage = ReplitObjectEmbeddingStorage(config)
        except Exception as e:
            logging.warning(f"Could not initialize Object Storage: {e}")
            return {}
        
        # Get embeddings and thread map from Object Storage
        try:
            # Directly use the embedding_storage's get_embeddings method
            # This avoids any circular imports with embedding_ops
            embeddings_array, thread_id_map = await embedding_storage.get_embeddings()
            
            if embeddings_array is None or thread_id_map is None:
                logging.warning("Could not load embeddings or thread ID map from Object Storage")
                return {}
                
            # Convert thread IDs to strings for consistent comparison
            thread_id_map = {str(k).strip(): v for k, v in thread_id_map.items()}
            thread_ids = [str(tid).strip() for tid in thread_ids]
            
            # Create mapping of thread IDs to embeddings
            embeddings_dict = {}
            missing_ids = []
            invalid_indices = []
            
            for thread_id in thread_ids:
                if thread_id in thread_id_map:
                    idx = thread_id_map[thread_id]
                    if idx < len(embeddings_array):
                        embeddings_dict[thread_id] = embeddings_array[idx].tolist()
                    else:
                        invalid_indices.append(idx)
                else:
                    missing_ids.append(thread_id)
            
            # Log debugging information
            if missing_ids and len(missing_ids) < 10:
                logging.debug(f"Missing thread IDs: {missing_ids}")
            elif missing_ids:
                logging.debug(f"Missing {len(missing_ids)} thread IDs out of {len(thread_ids)}")
                
            if invalid_indices:
                logging.warning(f"Found {len(invalid_indices)} invalid indices in thread map")
                
            if embeddings_dict:
                logging.info(f"Successfully retrieved {len(embeddings_dict)} embeddings out of {len(thread_ids)} requested")
            else:
                logging.warning("No embeddings found for the requested thread IDs")
                    
            return embeddings_dict
        except Exception as e:
            logging.error(f"Error retrieving embeddings from storage: {e}")
            logging.debug(traceback.format_exc())
            return {}
            
    except Exception as e:
        logging.error(f"Error fetching embeddings: {e}")
        logging.error(traceback.format_exc())
        return {}

async def save_embeddings_to_numpy(
    embeddings_data: Dict[str, List[float]],
    file_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """Save embeddings to Object Storage.
    
    Args:
        embeddings_data: Dictionary mapping thread IDs to embeddings
        file_path: Path to save the NumPy file (used for logging only)
        logger: Optional logger instance
        
    Returns:
        Path to the saved NumPy file
    """
    import numpy as np
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Verify input data is valid
        if not embeddings_data:
            logger.warning("No embeddings data provided, skipping save operation")
            return file_path
        
        # Convert embeddings to numpy array
        thread_ids = list(embeddings_data.keys())
        embeddings_list = []
        
        # Validate embeddings structure
        for tid in thread_ids:
            embedding = embeddings_data.get(tid)
            if embedding and isinstance(embedding, (list, np.ndarray)) and len(embedding) > 0:
                embeddings_list.append(embedding)
            else:
                logger.warning(f"Invalid embedding for thread {tid}, skipping")
                
        if not embeddings_list:
            logger.warning("No valid embeddings to save, skipping")
            return file_path
            
        # Convert to numpy arrays
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Ensure embeddings directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # First save to local NumPy file for backup
        try:
            # Save as local NumPy file
            np.savez_compressed(file_path, embeddings=embeddings_array)
            logger.info(f"Saved embeddings to local file: {file_path}")
        except Exception as local_save_error:
            logger.warning(f"Could not save embeddings to local file: {local_save_error}")
        
        # Create a clean thread ID map (valid thread IDs only)
        valid_thread_ids = [tid for i, tid in enumerate(thread_ids) if i < len(embeddings_list)]
        thread_id_map = {str(tid): idx for idx, tid in enumerate(valid_thread_ids)}
        
        try:
            # Initialize Object Storage
            from config.storage import ReplitObjectEmbeddingStorage
            from config.chanscope_config import ChanScopeConfig
            
            # Initialize storage
            config = ChanScopeConfig.from_env()
            embedding_storage = ReplitObjectEmbeddingStorage(config)
            
            # Store embeddings in Object Storage
            logger.info(f"Attempting to save {len(embeddings_array)} embeddings to Object Storage")
            success = await embedding_storage.store_embeddings(embeddings_array, thread_id_map)
            
            if success:
                logger.info(f"Successfully saved embeddings to Object Storage with shape {embeddings_array.shape}")
            else:
                logger.warning("Object Storage save operation returned False, but local file was saved")
                
        except ImportError as import_error:
            logger.warning(f"Object Storage modules not available: {import_error}. Using local file only.")
        except Exception as storage_error:
            logger.warning(f"Error saving to Object Storage: {storage_error}. Using local file only.")
            logger.debug(traceback.format_exc())
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        logger.error(traceback.format_exc())
        
        # Try a simpler approach as fallback - just save the file directly without Object Storage
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert raw data to numpy and save directly
            all_embeddings = list(embeddings_data.values())
            if all_embeddings:
                np.savez_compressed(file_path, embeddings=np.array(all_embeddings, dtype=np.float32))
                logger.info(f"Used fallback method to save embeddings to {file_path}")
                return file_path
        except Exception as fallback_error:
            logger.error(f"Fallback save also failed: {fallback_error}")
        
        # Return the original path even if save failed
        return file_path
    
    
def validate_text(text: str) -> Tuple[bool, str]:
    """Validate if text is suitable for processing and embedding.
    
    Args:
        text: The text to validate

    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    if text is None:
        return False, "Text is None"

    if not isinstance(text, str):
        return False, f"Text is not a string (type: {type(text)})"

    # Remove whitespace to check if there's actual content
    text_clean = text.strip()

    if not text_clean:
        return False, "Text is empty or whitespace only"

    if len(text_clean) < 10:
        return False, f"Text is too short ({len(text_clean)} chars)"

    # Check for excessive repetition which might confuse embedding models
    if len(set(text_clean)) / len(text_clean) < 0.1:
        return False, "Text has low character diversity (possible repetition)"

    return True, "Valid"
    
def get_venice_character_slug(character_slug: Optional[str] = None) -> str:
    """Get the Venice character slug from environment or config.
    
    Args:
        character_slug: Optional custom character slug to use
        
    Returns:
        Character slug to use with Venice API
    """
    # Priority 1: Use explicitly provided slug if given
    if character_slug:
        return character_slug
        
    # Priority 2: Check environment variable
    env_slug = os.environ.get("VENICE_CHARACTER_SLUG")
    if env_slug:
        return env_slug
    
    # Priority 3: Check Config class if available
    try:
        from config.settings import Config
        config_slug = Config.get_venice_character_slug()
        if config_slug:
            return config_slug
    except (ImportError, AttributeError):
        logger.debug("Could not import Config or get_venice_character_slug method not found")
        
    # Priority 4: Use default
    return "pisagor-ai"  # Default character

async def clean_logs(log_dir: Union[str, Path] = None) -> bool:
    """Clean log files on startup.
    
    Args:
        log_dir: Directory containing log files to clean
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Set default log directory if not provided
        if log_dir is None:
            try:
                from config.settings import Config
                log_dir = Path(Config.get_paths().get('logs_path', 'logs'))
            except Exception as e:
                logger.warning(f"Error getting log directory from Config: {e}")
                log_dir = Path('logs')
        
        # Convert string path to Path object if needed
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        
        # Create logs directory if it doesn't exist
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create logs directory {log_dir}: {e}")
            return False
        
        # Get list of log files
        log_files = list(log_dir.glob('*.log'))
        
        if not log_files:
            logger.info(f"No log files found in {log_dir}")
            return True
        
        logger.info(f"Cleaning {len(log_files)} log files in {log_dir}")
        
        # Clear each log file
        for log_file in log_files:
            try:
                # Truncate file to 0 bytes
                with open(log_file, 'w') as f:
                    pass
                logger.info(f"Cleared log file: {log_file}")
            except Exception as e:
                logger.warning(f"Could not clear log file {log_file}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning logs: {e}")
        logger.error(traceback.format_exc())
        return False

