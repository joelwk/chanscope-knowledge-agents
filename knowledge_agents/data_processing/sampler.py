import pandas as pd
from typing import Optional
from config.settings import Config
import random
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Sampler:
    """Class to handle data sampling operations."""
    
    def __init__(
        self,
        time_column: Optional[str] = None,
        strata_column: Optional[str] = None,
        initial_sample_size: Optional[int] = None,
        filter_date: Optional[str] = None):
        
        """Initialize sampler with configuration."""
        # Get configuration settings
        column_settings = Config.get_column_settings()
        sample_settings = Config.get_sample_settings()
        processing_settings = Config.get_processing_settings()
        
        self.time_column = time_column or column_settings['time_column']
        self.strata_column = strata_column or column_settings['strata_column']
        self.freq = processing_settings.get('freq', 'H')  # Default to hourly if not specified
        self.initial_sample_size = initial_sample_size or sample_settings['default_sample_size']
        
        # Handle filter date using Config's date parser
        filter_date = filter_date or processing_settings.get('filter_date')
        self.filter_date = Config._parse_date(filter_date) if filter_date else None
        if self.filter_date:
            logger.info(f"Using filter date: {self.filter_date} UTC")

    def filter_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize datetime column and apply filtering."""
        if self.time_column not in df.columns:
            logger.warning(f"Time column {self.time_column} missing from data")
            return df
            
        # Convert to UTC and coerce errors
        df[self.time_column] = pd.to_datetime(
            df[self.time_column], 
            utc=True,
            errors='coerce'
        )
        
        # Drop rows with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=[self.time_column])
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with invalid dates")

        return self.filter_by_date(df)

    def filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on a given date threshold."""
        if not self.filter_date or not self.time_column or self.time_column not in df.columns:
            return df
            
        # Convert filter_date to UTC datetime for comparison
        filter_datetime = pd.to_datetime(self.filter_date, utc=True)
        filtered_df = df[df[self.time_column] >= filter_datetime]
        return filtered_df
        
    def stratified_sample(self, data):
        """Main stratified sampling method."""
        data = self.filter_and_standardize(data)
        if self.time_column is not None and self.strata_column is not None and self.strata_column != "None":
            return self.sample_by_time_and_strata(data)
        elif self.time_column is not None:
            return self.sample_by_time(data)
        elif self.strata_column is not None and self.strata_column != "None":
            return self.sample_by_strata(data)
        else:
            return self.reservoir_sampling(data, self.initial_sample_size)

    def sample_by_time_and_strata(self, data):
        """Stratify data by time and strata."""
        data = data.copy()
        if self.time_column is None or self.strata_column is None or self.strata_column == "None":
            raise ValueError("Both time_column and strata_column must be provided for this method.")
            
        data.loc[:, 'temp_time_column'] = pd.to_datetime(data[self.time_column], utc=True, errors='coerce')
        time_samples = data.groupby(pd.Grouper(key='temp_time_column', freq=self.freq))
        samples = []

        for _, group in time_samples:
            if not group.empty:
                strata_samples = self.sample_by_strata(group, use_reservoir=False)
                samples.append(strata_samples)

        sampled_data = pd.concat(samples)
        sampled_data.drop(columns=['temp_time_column'], inplace=True)

        if len(sampled_data) > self.initial_sample_size:
            sampled_data = self.reservoir_sampling(sampled_data, self.initial_sample_size)

        return sampled_data

    def sample_by_time(self, data):
        """Sample data by time only."""
        data = data.copy()
        data.loc[:, 'temp_time_column'] = pd.to_datetime(data[self.time_column], utc=True, errors='coerce')
        sampled_data = data.groupby(pd.Grouper(key='temp_time_column', freq=self.freq)).apply(
            lambda x: x.sample(frac=min(1, int(self.initial_sample_size) / len(data))) if len(x) > 0 else x
        )
        sampled_data.reset_index(drop=True, inplace=True)
        sampled_data.drop(columns=['temp_time_column'], inplace=True)

        if len(sampled_data) > int(self.initial_sample_size):
            sampled_data = self.reservoir_sampling(sampled_data, int(self.initial_sample_size))

        return sampled_data
    
    def sample_by_strata(self, data, use_reservoir=True):
        """Sample data by strata only."""
        if self.strata_column is None or self.strata_column not in data.columns:
            raise ValueError(f"Strata column '{self.strata_column}' is not provided or not found in data.")

        # Reset index to avoid RangeIndex issues
        data = data.reset_index(drop=True)
        
        strata_values = data[self.strata_column].unique()
        strata_sample_size = self.initial_sample_size // len(strata_values)
        samples = []
        
        for value in strata_values:
            stratum_data = data[data[self.strata_column] == value]
            if len(stratum_data) > 0:
                sample_size = min(len(stratum_data), strata_sample_size)
                stratum_sample = stratum_data.sample(n=sample_size)
                samples.append(stratum_sample)
        
        sampled_data = pd.concat(samples, ignore_index=True)

        if use_reservoir and len(sampled_data) > int(self.initial_sample_size):
            sampled_data = self.reservoir_sampling(sampled_data, int(self.initial_sample_size))

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