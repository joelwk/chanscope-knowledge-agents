import pandas as pd
import random
import os
import logging
import warnings
from dotenv import load_dotenv

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

class Sampler:
    def __init__(self, filter_date: str, time_column: str = 'posted_date_time', strata_column: str = None, initial_sample_size: int = 1000):
        """Initialize the Sampler with configuration parameters."""
        # Prioritize the passed parameter over config values
        self.time_column = time_column if time_column is not None else os.getenv('TIME_COLUMN')
        self.strata_column = strata_column if strata_column is not None else os.getenv('STRATA_COLUMN')
        print(f"Time column: {self.time_column}, Strata column: {self.strata_column}")
        self.freq = os.getenv('FREQ', 'H')
        self.initial_sample_size = initial_sample_size if initial_sample_size is not None else int(os.getenv('INITIAL_SAMPLE_SIZE', '1000'))

        # Prioritize the passed filter_date parameter over config values
        config_filter_date = os.getenv('FILTER_DATE', None)
        if filter_date:
            self.filter_date = pd.to_datetime(filter_date, errors='coerce').date()
        elif config_filter_date:
            self.filter_date = pd.to_datetime(config_filter_date, errors='coerce').date()
        else:
            self.filter_date = None

    def filter_and_standardize(self, df):
        """Filter data by date and standardize time column if provided."""
        if self.time_column:
            df = self.filter_by_date(df)
            df.loc[:, self.time_column] = pd.to_datetime(df[self.time_column].str.replace(r'\+00:00', '', regex=True), errors='coerce')
        return df

    def filter_by_date(self, df):
        """Filter data based on a given date threshold."""
        if self.filter_date and self.time_column in df.columns:
            df['date'] = pd.to_datetime(df[self.time_column])
            df = df[df['date'].dt.date > self.filter_date]
        return df
        
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
            
        data.loc[:, 'temp_time_column'] = pd.to_datetime(data[self.time_column], errors='coerce')
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
        data.loc[:, 'temp_time_column'] = pd.to_datetime(data[self.time_column], errors='coerce')
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