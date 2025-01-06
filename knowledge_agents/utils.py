import os
import pandas as pd
import logging
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

def setup_logging(log_level=logging.INFO, log_file='generic_log.log'):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a')
        ]
    )
    
setup_logging(log_file='utility_func.log')

def safe_str_to_date(date_str, format="%Y-%m-%d %H:%M:%S"):
    try:
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, format)
    except (ValueError, TypeError) as e:
        print(f"Error converting '{date_str}' to date format '{format}':", e)
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
            df[col] = pd.to_datetime(df[col], errors='coerce')
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
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce') + pd.Timedelta(days=days)
        return df

    def subtract_days(self, df: pd.DataFrame, date_column: str, days: int) -> pd.DataFrame:
        """
        Subtract a specified number of days from a datetime column.
        """
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce') - pd.Timedelta(days=days)
        return df

    def filter_date_range(self, df: pd.DataFrame, date_column: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter rows based on a date range.
        """
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
        
    def format_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date_hour' in df.columns:
            df['date_hour'] = pd.to_datetime(df['date_hour'], errors='coerce')
            df['date_hour'] = df['date_hour'].dt.floor('H').dt.strftime('%Y-%m-%d %H')
        elif 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Timestamp'] = df['Timestamp'].dt.floor('H')
            df['date_hour'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H')
        else:
            raise ValueError("No 'date_hour' or 'Timestamp' column found in DataFrame.")
        return df
    
    def to_string(self, timestamp, format='%Y-%m-%d %H:%M:%S'):
        return timestamp.strftime(format) if pd.notnull(timestamp) else None