import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
from .cloud_handler import S3Handler, load_all_csv_data_from_s3
from ..data_ops import DataConfig, DataOperations

logger = logging.getLogger(__name__)

class DataScheduler:
    def __init__(self, config: DataConfig):
        self.config = config
        logger.info("Initializing DataScheduler with config: %s", config)
        self.s3_handler = S3Handler()
        self.data_ops = DataOperations(config)
        
    async def initialize_storage(self):
        """Initial load of the last week's data."""
        one_week_ago = datetime.now(pytz.UTC) - timedelta(days=7)
        self.config.filter_date = one_week_ago.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Initializing storage with data from: {self.config.filter_date}")
        
        # Load fresh data from S3
        all_data = []
        for chunk in load_all_csv_data_from_s3(self.config.filter_date):
            all_data.append(chunk)
            
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df.to_csv(self.config.all_data_path, index=False)
            logger.info(f"Loaded {len(df)} records from S3")
            
            # Process the data
            await self.data_ops._process_existing_data()
            logger.info("Storage initialization completed")
        else:
            logger.warning("No data found in S3 for the specified date range")
        
    async def update_storage(self):
        """Hourly update of data."""
        try:
            logger.info("Starting storage update process")
            current_time = datetime.now(pytz.UTC)
            cutoff_time = current_time - timedelta(days=7)
            logger.info(f"Update window: {cutoff_time} to {current_time}")
            
            # Set filter date to last successful update or one hour ago
            one_hour_ago = current_time - timedelta(hours=1)
            self.config.filter_date = one_hour_ago.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Set filter date to: {self.config.filter_date}")
            
            # Load new data from S3
            new_data = []
            for chunk in load_all_csv_data_from_s3(self.config.filter_date):
                new_data.append(chunk)
                
            if new_data:
                new_df = pd.concat(new_data, ignore_index=True)
                logger.info(f"Found {len(new_df)} new records")
                
                # Read and update existing data
                logger.info("Reading existing data...")
                if self.config.all_data_path.exists():
                    df = pd.read_csv(self.config.all_data_path)
                    df['posted_date_time'] = pd.to_datetime(df['posted_date_time'], utc=True)
                    
                    # Remove outdated records
                    original_len = len(df)
                    df = df[df['posted_date_time'] >= cutoff_time]
                    filtered_len = len(df)
                    logger.info(f"Removed {original_len - filtered_len} outdated records")
                    
                    # Append new data
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=['thread_id', 'posted_date_time'])
                else:
                    df = new_df
                
                # Save updated dataset
                logger.info("Saving updated dataset...")
                df.to_csv(self.config.all_data_path, index=False)
                
                # Update stratified data and knowledge base
                logger.info("Updating stratified data and knowledge base...")
                await self.data_ops._process_existing_data()
                
                logger.info(f"Storage update completed. Data range: {df['posted_date_time'].min()} to {df['posted_date_time'].max()}")
            else:
                logger.info("No new data found for the update period")
            
        except Exception as e:
            logger.error(f"Error updating storage: {str(e)}", exc_info=True)
            raise
