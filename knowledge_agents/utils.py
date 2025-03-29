import os
import pandas as pd
import logging
import warnings
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple, List
import traceback

# Import centralized logging configuration
from config.logging_config import get_logger
from knowledge_agents.embedding_ops import load_embeddings, load_thread_id_map

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

def save_query_output(
    response: Dict[str, Any], 
    base_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    include_embeddings: bool = True,
    save_numpy: bool = True,
    query: Optional[str] = None
) -> Tuple[Path, Optional[Path]]:
    """Save query output to JSON files with embeddings in separate files.
    
    Args:
        response: Query response dictionary
        base_path: Optional base path for saving files
        logger: Optional logger instance
        include_embeddings: Whether to include embeddings in the output
        save_numpy: Whether to save embeddings in NumPy format (.npz) in addition to JSON
        query: The original query text to include in the output
        
    Returns:
        Tuple of (json_path, embeddings_path) for the saved files
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Ensure we have a base path
    if base_path is None:
        base_path = Path("generated_data")
    else:
        base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings directory
    embeddings_dir = base_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames with more precision to ensure uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Prepare paths
    json_path = base_path / f"query_output_{timestamp}.json"
    embeddings_json_path = embeddings_dir / f"query_output_{timestamp}_embeddings.json"
    embeddings_npy_path = embeddings_dir / f"query_output_{timestamp}_embeddings.npz"
    
    try:
        # Create a copy of the response without embeddings for the main JSON file
        response_without_embeddings = response.copy()
        embeddings_data = {}
        
        # Add the original query to the main response if provided
        if query and "query" not in response_without_embeddings:
            response_without_embeddings["query"] = query
        
        # If embeddings are requested, fetch them for each thread
        if include_embeddings:
            thread_ids = [chunk["thread_id"] for chunk in response.get("chunks", [])]
            thread_embeddings = _fetch_embeddings_for_threads(thread_ids)
            
            # Add embeddings to response if found
            if thread_embeddings:
                for chunk in response["chunks"]:
                    thread_id = chunk["thread_id"]
                    if thread_id in thread_embeddings:
                        # Store embeddings in the separate data structure
                        embeddings_data[thread_id] = thread_embeddings[thread_id]
                        
                        # Add a reference to the embeddings file in the main response
                        if "chunks" in response_without_embeddings:
                            for chunk_without_embedding in response_without_embeddings["chunks"]:
                                if chunk_without_embedding["thread_id"] == thread_id:
                                    # Add references to both JSON and NPY files if applicable
                                    embedding_refs = {
                                        "json": str(embeddings_json_path)
                                    }
                                    if save_numpy:
                                        embedding_refs["numpy"] = str(embeddings_npy_path)
                                    chunk_without_embedding["embedding_references"] = embedding_refs
        
        # Save main JSON output without embeddings
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(response_without_embeddings, f, indent=2, ensure_ascii=False)
        
        # Save embeddings to separate files if we have any
        if embeddings_data and include_embeddings:
            # Save as JSON
            with open(embeddings_json_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
            
            # Save as NumPy if requested
            if save_numpy:
                save_embeddings_to_numpy(embeddings_data, embeddings_npy_path, logger)
            
            logger.info(f"Saved query output to {json_path} and embeddings to {embeddings_json_path}")
            if save_numpy:
                logger.info(f"Also saved embeddings in NumPy format to {embeddings_npy_path}")
            
            return json_path, embeddings_json_path
        else:
            logger.info(f"Saved query output to {json_path}")
            return json_path, None
        
    except Exception as e:
        logger.error(f"Error saving query output: {e}")
        logger.error(traceback.format_exc())
        raise

def _fetch_embeddings_for_threads(thread_ids: List[str]) -> Dict[str, List[float]]:
    """Fetch embeddings for a list of thread IDs.
    
    Args:
        thread_ids: List of thread IDs to fetch embeddings for
        
    Returns:
        Dictionary mapping thread IDs to their embeddings
    """
    try:
        # Initialize paths
        data_dir = Path("data")
        stratified_dir = data_dir / 'stratified'
        embeddings_path = stratified_dir / 'embeddings.npz'
        thread_id_map_path = stratified_dir / 'thread_id_map.json'
        
        # Load embeddings and thread ID map
        embeddings_array, metadata = load_embeddings(embeddings_path)
        thread_id_map = load_thread_id_map(thread_id_map_path)
        
        if embeddings_array is None or thread_id_map is None:
            logging.warning("Could not load embeddings or thread ID map")
            return {}
            
        # Convert thread IDs to strings for consistent comparison
        thread_id_map = {str(k): v for k, v in thread_id_map.items()}
        thread_ids = [str(tid) for tid in thread_ids]
        
        # Create mapping of thread IDs to embeddings
        embeddings_dict = {}
        for thread_id in thread_ids:
            if thread_id in thread_id_map:
                idx = thread_id_map[thread_id]
                if idx < len(embeddings_array):
                    embeddings_dict[thread_id] = embeddings_array[idx].tolist()
                    
        return embeddings_dict
        
    except Exception as e:
        logging.error(f"Error fetching embeddings: {e}")
        logging.error(traceback.format_exc())
        return {}

def save_embeddings_to_numpy(
    embeddings_data: Dict[str, List[float]],
    file_path: Path,
    logger: Optional[logging.Logger] = None
) -> Path:
    """Save embeddings to a NumPy file.
    
    Args:
        embeddings_data: Dictionary mapping thread IDs to embeddings
        file_path: Path to save the NumPy file
        logger: Optional logger instance
        
    Returns:
        Path to the saved NumPy file
    """
    import numpy as np
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Convert embeddings to numpy array
        thread_ids = list(embeddings_data.keys())
        embeddings_list = [embeddings_data[tid] for tid in thread_ids]
        
        # Save as .npz file with thread_ids as keys
        np.savez(
            file_path, 
            thread_ids=np.array(thread_ids, dtype=str),
            embeddings=np.array(embeddings_list, dtype=np.float32)
        )
        
        logger.info(f"Saved embeddings to NumPy file: {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving embeddings to NumPy file: {e}")
        logger.error(traceback.format_exc())
        raise