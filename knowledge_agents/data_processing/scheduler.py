import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
from .cloud_handler import S3Handler, load_all_csv_data_from_s3
from ..data_ops import DataConfig, prepare_knowledge_base
from config.settings import Config

logger = logging.getLogger(__name__)

class DataScheduler:
    """Handles scheduled data processing operations."""

    def __init__(self, config: DataConfig):
        """Initialize scheduler with configuration."""
        self.config = config
        logger.info(f"Initialized DataScheduler with config: {config}")
        self.s3_handler = S3Handler()

    async def initialize_storage(self):
        """Initial load of the filtered date data."""
        try:
            # Set initial time window
            logger.info(f"Initializing storage with data from: {self.config.filter_date}")

            # Load fresh data from S3
            total_records = 0
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=self.config.filter_date):
                total_records += len(chunk_df)
                del chunk_df  # Clean up chunk after counting

            logger.info(f"Loaded {total_records} records from S3")

            if total_records > 0:
                # Process data using the full pipeline
                try:
                    result = await prepare_knowledge_base(force_refresh=True)
                    logger.info(f"Knowledge base preparation completed: {result}")
                except Exception as e:
                    logger.error(f"Error preparing knowledge base: {e}")
                    raise
            else:
                logger.warning("No data found in S3 for the specified date range")

        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            raise

    async def update_storage(self):
        """Hourly update of data."""
        try:
            logger.info("Starting storage update process")

            # Set up time window for update
            current_time = datetime.now(pytz.UTC)
            cutoff_time = current_time - timedelta(days=7)
            one_hour_ago = current_time - timedelta(hours=1)
            self.config.filter_date = one_hour_ago.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Set filter date to: {self.config.filter_date}")

            # Check for new data
            total_new_records = 0
            for chunk_df in load_all_csv_data_from_s3(latest_date_processed=self.config.filter_date):
                total_new_records += len(chunk_df)
                del chunk_df
            logger.info(f"Update window: {cutoff_time} to {current_time}")

            # Set filter date to last successful update or one hour ago

            if total_new_records > 0:
                logger.info(f"Found {total_new_records} new records")
                # Process new data using the full pipeline
                # force_refresh=False will handle merging with existing data
                result = await prepare_knowledge_base(force_refresh=False)
                logger.info(f"Knowledge base update completed: {result}")
            else:
                logger.info("No new data found for the update period")

        except Exception as e:
            logger.error(f"Storage update failed: {e}")
            raise
