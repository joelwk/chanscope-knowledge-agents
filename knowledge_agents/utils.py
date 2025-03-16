import os
import pandas as pd
import logging
import warnings
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, Optional

# Import centralized logging configuration
from config.logging_config import get_logger

# Create a logger using the centralized configuration
logger = get_logger('knowledge_agents.utils')

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_logger(name, level=logging.INFO):
    """Get a logger with the specified name and level.
    
    This is a wrapper around the centralized logging configuration.
    
    Args:
        name: The name of the logger
        level: The logging level
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

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
        "%d/%m/%Y"
    ]
    
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
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """Save query response data and embeddings to organized directory structure.
    
    Args:
        response: Query response dictionary containing chunks, summary, and metadata
        base_path: Optional base path override (defaults to data/generated_data)
        logger: Optional logger instance
        
    Returns:
        Dictionary with paths where files were saved
        
    Raises:
        ValueError: If response is invalid or required paths cannot be created
        IOError: If there are issues writing the files
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        # Validate response
        if not isinstance(response, dict):
            raise ValueError("Response must be a dictionary")
            
        # Set up paths with proper error handling
        if base_path is None:
            from config.settings import Config
            try:
                paths = Config.get_paths()
                base_path = paths.get('generated_data')
                if base_path is None:
                    raise ValueError("Could not determine generated_data path from config")
            except Exception as e:
                logger.error(f"Error getting base path from config: {e}")
                base_path = "data/generated_data"  # Fallback default
                
        base_dir = Path(base_path)
        logger.info(f"Using base directory: {base_dir}")
        embeddings_dir = base_dir / "embeddings"
        
        # Create directories with error handling
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            embeddings_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IOError(f"Failed to create required directories: {e}")
        
        # Extract temporal context for filename
        temporal_context = response.get("metadata", {}).get("temporal_context", {})
        end_date = temporal_context.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        
        # Generate base filename using end date
        base_filename = f"query_output_{end_date}"
        
        # Save main response data as JSON
        response_path = base_dir / f"{base_filename}.json"
        with open(response_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
            
        # Save text data (chunks and summary) separately for easier access
        text_data = {
            "summary": response.get("summary", ""),
            "chunks": [
                {
                    "thread_id": chunk.get("thread_id", "unknown"),
                    "posted_date_time": chunk.get("posted_date_time", "unknown"),
                    "analysis": chunk.get("analysis", {}).get("thread_analysis", "")
                }
                for chunk in response.get("chunks", [])
            ]
        }
        text_path = base_dir / f"{base_filename}_text.json"
        with open(text_path, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
            
        # Save embeddings if present in chunks
        embeddings_data = {}
        embeddings_path = None
        
        # Extract embeddings from chunks
        for chunk in response.get("chunks", []):
            if "embedding" in chunk and chunk["embedding"] is not None:
                thread_id = chunk.get("thread_id", "unknown")
                embeddings_data[thread_id] = chunk["embedding"]

        # Save embeddings if any were found
        if embeddings_data:
            embeddings_path = embeddings_dir / f"{base_filename}_embeddings.npz"
            # Save embeddings with thread IDs for reference
            np.savez_compressed(
                embeddings_path,
                embeddings=np.array(list(embeddings_data.values())),
                thread_ids=np.array(list(embeddings_data.keys())),
                metadata=np.array(json.dumps({
                    "date": end_date,
                    "count": len(embeddings_data),
                    "dimensions": len(next(iter(embeddings_data.values())))
                }).encode())
            )
        
        saved_paths = {
            "response": str(response_path),
            "text": str(text_path),
            "embeddings": str(embeddings_path) if embeddings_data else None
        }
        
        logger.info(f"Saved query output to: {saved_paths}")
        return saved_paths
        
    except Exception as e:
        logger.error(f"Error saving query output: {str(e)}")
        raise