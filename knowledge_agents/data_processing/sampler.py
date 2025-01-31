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
        self.time_column = time_column or Config.TIME_COLUMN
        self.strata_column = strata_column or Config.STRATA_COLUMN
        self.freq = Config.FREQ if hasattr(Config, 'FREQ') else 'H'
        self.initial_sample_size = initial_sample_size or Config.DEFAULT_SAMPLE_SIZE
        
        # Handle filter date - convert to datetime once during initialization
        config_filter_date = Config.FILTER_DATE if hasattr(Config, 'FILTER_DATE') else None
        filter_date = filter_date or config_filter_date
        
        if filter_date:
            try:
                self.filter_date = pd.to_datetime(filter_date, utc=True)
            except (ValueError, TypeError):
                logger.warning(f"Invalid filter date format: {filter_date}, setting to None")
                self.filter_date = None
        else:
            self.filter_date = None

    def filter_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data by date and standardize time column if provided."""
        if not self.time_column or self.time_column not in df.columns:
            return df
            
        # Standardize datetime column first
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column], utc=True, errors='coerce')
        
        # Then apply date filtering
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

        strata_values = data[self.strata_column].unique()
        strata_sample_size = self.initial_sample_size // len(strata_values)
        samples = [
            data[data[self.strata_column] == value].sample(
                min(len(data[data[self.strata_column] == value]), strata_sample_size)
            ) for value in strata_values
        ]
        sampled_data = pd.concat(samples)

        if use_reservoir and len(sampled_data) > int(self.initial_sample_size):
            sampled_data = self.reservoir_sampling(sampled_data, int(self.initial_sample_size))

        return sampled_data

    def reservoir_sampling(self, data, k):
        """Perform reservoir sampling on the data."""
        reservoir = []
        for i, row in enumerate(data.iterrows()):
            if i < k:
                reservoir.append(row[1])
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = row[1]
        return pd.DataFrame(reservoir)