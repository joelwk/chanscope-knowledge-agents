import boto3
import os
import pandas as pd
import logging
import gc
from pathlib import Path
from typing import Optional
from dateutil import tz
from config.settings import Config
from io import StringIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class S3Handler:
    """Handler for S3 operations."""

    def __init__(self, local_data_path: str = '.'):
        """Initialize S3 handler with configuration."""
        self.bucket_name = Config.S3_BUCKET
        self.bucket_prefix = Config.S3_BUCKET_PREFIX
        self.local_path = Path(local_data_path).resolve()
        self.region_name = Config.AWS_DEFAULT_REGION
        self.aws_access_key_id = Config.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = Config.AWS_SECRET_ACCESS_KEY
        self.s3 = self._create_s3_client()

    def _create_s3_client(self):
        """Create an S3 client with credentials from Config."""
        session = boto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
        return session.client('s3')

    def get_s3_client(self):
        return self.s3

    def file_exists_in_s3(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def upload_dir(self, dir_key: str):
        """Upload a directory to S3."""
        for root, _, files in os.walk(self.local_path):
            for file in files:
                local_file = Path(root) / file
                relative_path = local_file.relative_to(self.local_path)
                s3_key = str(Path(dir_key) / relative_path)

                if not s3_key.endswith('/'):
                    if not self.file_exists_in_s3(s3_key):
                        self._upload_file(str(local_file), s3_key)
                    else:
                        logger.info(f"Skipping {s3_key}, already exists in S3.")

    def download_dir(self, dir_key: str, local_file_path: Optional[str] = None):
        """Download a directory from S3."""
        if not local_file_path:
            local_file_path = self.local_path / Path(dir_key).name
        local_file_path = Path(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=dir_key):
            for file in result.get('Contents', []):
                self.download_file(file['Key'])

    def download_file(self, s3_key: str, local_file_path: Optional[str] = None):
        """Download a file from S3."""
        if not local_file_path:
            local_file_path = self.local_path / Path(s3_key).name
        local_file_path = Path(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.s3.download_file(self.bucket_name, s3_key, str(local_file_path))
        logger.info(f"Downloaded file {s3_key} to {local_file_path}")

    def _upload_file(self, local_file: str, key: str):
        """Upload a file to S3."""
        self.s3.upload_file(local_file, self.bucket_name, key)
        logger.info(f"Uploaded file {local_file} to {self.bucket_name}/{key}")

    def stream_csv_data(self, latest_date_processed: str = None):
        """Stream CSV data from S3, filtering by LastModified date.

        Args:
            latest_date_processed: Date string in YYYY-MM-DD format to filter data

        Yields:
            pd.DataFrame: Chunks of filtered data
        """
        try:
            logger.info("Streaming data from S3 bucket: %s", self.bucket_name)
            logger.info(f"Initial filter date: {latest_date_processed}")

            # Convert latest_date_processed to UTC datetime for metadata comparison
            latest_date = None
            if latest_date_processed:
                try:
                    # Parse the date in YYYY-MM-DD format and set to start of day in UTC
                    latest_date = pd.to_datetime(latest_date_processed).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    if latest_date.tzinfo is None:
                        latest_date = latest_date.tz_localize('UTC')
                    logger.info(f"Using filter date: {latest_date} UTC")
                except Exception as e:
                    logger.error(f"Error parsing date {latest_date_processed}. Expected format: YYYY-MM-DD")
                    logger.error(f"Error details: {str(e)}")
                    raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format (e.g., 2025-01-30)")

            # Use Config.SAMPLE_SIZE to determine chunk size
            chunk_size = min(5000, Config.SAMPLE_SIZE)  # Reduced chunk size for better control
            logger.info(f"Using chunk size: {chunk_size} (Sample size: {Config.SAMPLE_SIZE})")

            total_rows_processed = 0
            for s3_key in self._get_filtered_csv_files(latest_date):
                try:
                    temp_file = Path(f"temp_{Path(s3_key).name}")
                    try:
                        logger.info(f"Processing file: {s3_key}")
                        self.s3.download_file(self.bucket_name, s3_key, str(temp_file))

                        for chunk in pd.read_csv(
                            temp_file,
                            chunksize=chunk_size,
                            usecols=['thread_id', 'posted_date_time', 'text_clean', 'posted_comment'],
                            on_bad_lines='skip',
                            encoding='utf-8'
                        ):
                            try:
                                # Convert posted_date_time to datetime with explicit UTC handling
                                chunk['posted_date_time'] = pd.to_datetime(chunk['posted_date_time'], utc=True, errors='coerce')
                                chunk = chunk.dropna(subset=['posted_date_time'])

                                # Apply date filter with detailed logging
                                if latest_date is not None:
                                    before_filter = len(chunk)
                                    chunk = chunk[chunk['posted_date_time'] >= latest_date]
                                    after_filter = len(chunk)
                                    if before_filter != after_filter:
                                        logger.info(f"Date filter: {before_filter - after_filter} rows filtered out")
                                        logger.debug(f"Date range in chunk: {chunk['posted_date_time'].min()} to {chunk['posted_date_time'].max()}")

                                if not chunk.empty:
                                    total_rows_processed += len(chunk)
                                    if total_rows_processed > Config.SAMPLE_SIZE:
                                        logger.warning(f"Total processed rows ({total_rows_processed}) exceeds SAMPLE_SIZE ({Config.SAMPLE_SIZE})")
                                        # Calculate remaining rows needed
                                        rows_needed = Config.SAMPLE_SIZE - (total_rows_processed - len(chunk))
                                        if rows_needed > 0:
                                            logger.info(f"Taking only {rows_needed} rows from current chunk to meet sample size")
                                            yield chunk.head(rows_needed)
                                        logger.info("Stopping data stream as sample size limit reached")
                                        return
                                    yield chunk

                            except Exception as e:
                                logger.error(f"Error processing chunk from {s3_key}: {str(e)}")
                                continue

                        logger.info(f"Completed processing {s3_key}: {total_rows_processed} total rows processed")

                    except Exception as e:
                        logger.error(f"Error processing file {s3_key}: {str(e)}")
                    finally:
                        if temp_file.exists():
                            temp_file.unlink()

                except Exception as e:
                    logger.error(f"Error handling file {s3_key}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Critical error in stream_csv_data: {str(e)}")
            raise

    def _get_filtered_csv_files(self, latest_date):
        """Get filtered list of CSV files from S3."""
        logger.info(f"Filtering CSV files with date threshold: {latest_date}")
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name, 
            Prefix=self.bucket_prefix)

        if 'Contents' not in response:
            logger.warning(f"No files found in bucket {self.bucket_name}")
            return []

        csv_objects = []
        total_files = 0
        filtered_by_date = 0
        filtered_by_board = 0
        
        for item in response.get('Contents', []):
            if item['Key'].endswith('.csv'):
                total_files += 1
                file_date = item['LastModified'].astimezone(tz.UTC)
                logger.debug(f"Found CSV file: {item['Key']}, modified: {file_date}")
                
                # Check board filter
                board_match = Config.SELECT_BOARD is None or f'chanscope_{Config.SELECT_BOARD}' in item['Key']
                if not board_match:
                    filtered_by_board += 1
                    logger.debug(f"Skipping file due to board mismatch: {item['Key']}")
                    continue

                # Check date filter with strict comparison
                if latest_date is not None:
                    if file_date <= latest_date:
                        filtered_by_date += 1
                        logger.debug(f"Skipping file due to date: {item['Key']} ({file_date} <= {latest_date})")
                        continue
                    logger.info(f"Including file: {item['Key']} (modified: {file_date})")
                
                csv_objects.append(item['Key'])

        logger.info(f"File filtering summary:")
        logger.info(f"- Total CSV files found: {total_files}")
        logger.info(f"- Files filtered by date: {filtered_by_date}")
        logger.info(f"- Files filtered by board: {filtered_by_board}")
        logger.info(f"- Files selected for processing: {len(csv_objects)}")
        return csv_objects

def load_all_csv_data_from_s3(latest_date_processed: str = None):
    """Load all CSV data from S3 bucket using streaming and S3 Select.

    Yields:
        pd.DataFrame: Chunks of the filtered data.
    """
    connector = S3Handler()
    yield from connector.stream_csv_data(latest_date_processed)