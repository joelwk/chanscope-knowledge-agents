import pandas as pd
from typing import Optional
from config.settings import Config
import random
import logging
from config.config_utils import parse_filter_date

logger = logging.getLogger(__name__)

class Sampler:
    """Class to handle data sampling operations."""

    def __init__(
        self,
        time_column: Optional[str] = None,
        strata_column: Optional[str] = None,
        initial_sample_size: Optional[int] = None,
        filter_date: Optional[str] = None,
        window_hours: Optional[int] = 12):

        """Initialize sampler with configuration."""
        # Get configuration settings
        column_settings = Config.get_column_settings()
        sample_settings = Config.get_sample_settings()
        processing_settings = Config.get_processing_settings()

        self.time_column = time_column or column_settings['time_column']
        # Force strata_column to None to ensure time-based sampling only
        self.strata_column = None
        self.freq = processing_settings.get('freq', 'H')  # Default to hourly if not specified
        self.initial_sample_size = initial_sample_size or sample_settings['default_sample_size']

        # Handle filter date using Config's date parser
        filter_date = filter_date or processing_settings.get('filter_date')
        self.filter_date = parse_filter_date(filter_date) if filter_date else None
        if self.filter_date:
            logger.info(
                f"Using filter date: {self.filter_date} UTC ")

    def filter_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize datetime column and apply filtering."""
        if self.time_column not in df.columns:
            logger.warning(f"Time column {self.time_column} missing from data")
            return df

        # Convert to UTC and coerce errors
        df[self.time_column] = pd.to_datetime(
            df[self.time_column], 
            utc=True,
            errors='coerce')

        # Drop rows with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=[self.time_column])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with invalid dates")
        return self.filter_by_date(df)

    def filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to include records from the filter_date up to current time.
        This method filters data to include all records that occurred after
        the specified filter_date, up to the current time.
        """
        if not self.filter_date or not self.time_column or self.time_column not in df.columns:
            return df

        # Log available date range
        date_range = df[self.time_column].agg(['min', 'max'])
        logger.info(f"Dataset date range: from {date_range['min']} to {date_range['max']} UTC")

        # Convert filter_date to UTC datetime for comparison
        filter_datetime = pd.to_datetime(self.filter_date, utc=True)
        current_time = pd.Timestamp.now(tz='UTC')
        
        # Filter data from filter_date to current time
        filtered_df = df[
            (df[self.time_column] >= filter_datetime)]
        logger.info(
            f"Filtered data from {filter_datetime} UTC to {current_time} UTC. "
            f"Kept {len(filtered_df)}/{len(df)} records")
        return filtered_df

    def stratified_sample(self, data):
        """Main stratified sampling method."""
        data = self.filter_and_standardize(data)
        return self.sample_by_time(data)

    def sample_by_time(self, data):
        """Sample data by time only."""
        if len(data) == 0:
            return pd.DataFrame(columns=data.columns)
        data = data.copy()
        
        # First, apply reservoir sampling if data is too large
        if len(data) > self.initial_sample_size * 2:
            data = self.reservoir_sampling(data, self.initial_sample_size * 2)
        data.loc[:, 'temp_time_column'] = pd.to_datetime(data[self.time_column], utc=True, errors='coerce')
        
        # Group by time
        time_samples = data.groupby(pd.Grouper(key='temp_time_column', freq=self.freq))
        
        # Calculate target size per time group
        n_groups = sum(1 for _, group in time_samples if not group.empty)
        if n_groups == 0:
            return pd.DataFrame(columns=data.columns)
        target_size_per_group = max(5, self.initial_sample_size // n_groups)
        
        samples = []
        for _, group in time_samples:
            if not group.empty:
                if len(group) > target_size_per_group:
                    group = self.reservoir_sampling(group, target_size_per_group)
                samples.append(group)
        
        if not samples:
            return pd.DataFrame(columns=data.columns)
            
        sampled_data = pd.concat(samples, ignore_index=True)
        sampled_data.drop(columns=['temp_time_column'], inplace=True)

        # Final reservoir sampling to ensure we don't exceed target size
        if len(sampled_data) > self.initial_sample_size:
            sampled_data = self.reservoir_sampling(sampled_data, self.initial_sample_size)

        return sampled_data

    def reservoir_sampling(self, data, k):
        """Perform reservoir sampling on the data."""
        # Reset index to avoid RangeIndex issues
        data = data.reset_index(drop=True)
        reservoir = []
        for i, row in enumerate(data.itertuples()):
            if i < k:
                reservoir.append(row._asdict())
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = row._asdict()
        result = pd.DataFrame(reservoir)
        # Remove the Index column if it was added by _asdict()
        if 'Index' in result.columns:
            result = result.drop('Index', axis=1)
        return result
