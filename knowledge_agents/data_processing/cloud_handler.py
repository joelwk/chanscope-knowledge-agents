import boto3
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Generator
from dateutil import tz
from config.settings import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class S3Handler:
    """Handler for S3 operations."""

    def __init__(self, local_data_path: str = '.'):
        """Initialize S3 handler with configuration."""
        # Get AWS settings from Config
        aws_settings = Config.get_aws_settings()
        processing_settings = Config.get_processing_settings()
        paths = Config.get_paths()
        
        self.bucket_name = aws_settings['s3_bucket']
        self.bucket_prefix = aws_settings['s3_bucket_prefix']
        self.local_path = Path(paths['root_data_path']).resolve()
        self.region_name = aws_settings['aws_default_region']
        self.aws_access_key_id = aws_settings['aws_access_key_id']
        self.aws_secret_access_key = aws_settings['aws_secret_access_key']
        self.select_board = processing_settings.get('select_board')
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

    def stream_csv_data(self, filter_date: Optional[str] = None) -> Generator[pd.DataFrame, None, None]:
        """Stream CSV data from S3 in chunks."""
        try:
            # Get centralized settings
            chunk_settings = Config.get_chunk_settings()
            sample_settings = Config.get_sample_settings()
            column_settings = Config.get_column_settings()
            
            # Use sample size from centralized settings
            sample_size = sample_settings['default_sample_size']
            chunk_size = min(chunk_settings['default_chunk_size'], sample_size)
            logger.info(f"Using chunk size: {chunk_size} (Sample size: {sample_size})")

            # Parse filter date using centralized method
            latest_date = Config._parse_date(filter_date) if filter_date else None
            if latest_date:
                logger.info(f"Using filter date: {latest_date} UTC")

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
                            usecols=column_settings['required_columns'],
                            on_bad_lines='skip',
                            encoding='utf-8'
                        ):
                            try:
                                # Convert posted_date_time to datetime with explicit UTC handling
                                chunk[column_settings['time_column']] = pd.to_datetime(
                                    chunk[column_settings['time_column']], 
                                    utc=True, 
                                    errors='coerce'
                                )
                                chunk = chunk.dropna(subset=[column_settings['time_column']])

                                # Apply date filter with detailed logging
                                if latest_date is not None:
                                    before_filter = len(chunk)
                                    # Include a small buffer (1 day) to account for timezone differences
                                    filter_date_with_buffer = latest_date - pd.Timedelta(days=1)
                                    chunk = chunk[chunk[column_settings['time_column']] >= filter_date_with_buffer]
                                    after_filter = len(chunk)
                                    if before_filter != after_filter:
                                        logger.info(f"Date filter: {before_filter - after_filter} rows filtered out")
                                        logger.info(f"Date range in chunk: {chunk[column_settings['time_column']].min()} to {chunk[column_settings['time_column']].max()}")

                                if not chunk.empty:
                                    total_rows_processed += len(chunk)
                                    if total_rows_processed > sample_size:
                                        logger.warning(f"Total processed rows ({total_rows_processed}) exceeds sample size ({sample_size})")
                                        rows_needed = sample_size - (total_rows_processed - len(chunk))
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

    def _get_filtered_csv_files(self, latest_date: Optional[str] = None) -> list:
        """Get filtered list of CSV files from S3 bucket."""
        logger.info("=== Getting Filtered CSV Files ===")
        logger.info(f"Latest date filter: {latest_date}")
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.bucket_prefix
            )
            logger.info(f"S3 list_objects_v2 response received")
            
            if 'Contents' not in response:
                logger.warning(f"No files found in bucket {self.bucket_name} with prefix {self.bucket_prefix}")
                return []
            
            logger.info(f"Found {len(response['Contents'])} total objects in bucket")

            csv_objects = []
            total_files = 0
            filtered_by_date = 0
            filtered_by_board = 0
            
            for item in response.get('Contents', []):
                if item['Key'].endswith('.csv'):
                    total_files += 1
                    file_date = item['LastModified'].astimezone(tz.UTC)
                    logger.debug(f"Found CSV file: {item['Key']}, modified: {file_date}")
                    
                    # Check board filter using configuration
                    board_match = self.select_board is None or f'chanscope_{self.select_board}' in item['Key']
                    if not board_match:
                        filtered_by_board += 1
                        logger.debug(f"Skipping file due to board mismatch: {item['Key']}")
                        continue

                    # Check date filter with strict comparison
                    if latest_date is not None:
                        # Compare as timezone-aware datetimes
                        # We want files that might contain data after our filter date
                        # So we only filter out files that are definitely too old
                        # (modified more than 30 days before our filter date)
                        cutoff_date = latest_date - pd.Timedelta(days=30)
                        if file_date < cutoff_date:
                            filtered_by_date += 1
                            logger.debug(f"Skipping file due to date: {item['Key']} ({file_date} < {cutoff_date})")
                            continue
                        logger.info(f"Including file: {item['Key']} (modified: {file_date})")
                    
                    csv_objects.append(item['Key'])

            logger.info(f"File filtering summary:")
            logger.info(f"- Total objects in bucket: {len(response['Contents'])}")
            logger.info(f"- Total CSV files found: {total_files}")
            logger.info(f"- Files filtered by date: {filtered_by_date}")
            logger.info(f"- Files filtered by board: {filtered_by_board}")
            logger.info(f"- Files selected for processing: {len(csv_objects)}")
            
            if len(csv_objects) == 0:
                logger.warning("No files selected for processing after filtering")
            
            return csv_objects
            
        except Exception as e:
            logger.error(f"Error getting filtered CSV files: {str(e)}")
            raise

def load_all_csv_data_from_s3(latest_date_processed: str = None):
    """Load all CSV data from S3 bucket using streaming and S3 Select.

    Yields:
        pd.DataFrame: Chunks of the filtered data.
    """
    logger.info("=== Starting S3 Data Loading ===")
    logger.info(f"Latest date processed: {latest_date_processed}")
    
    try:
        connector = S3Handler()
        logger.info(f"S3 Configuration:")
        logger.info(f"- Bucket: {connector.bucket_name}")
        logger.info(f"- Prefix: {connector.bucket_prefix}")
        logger.info(f"- Region: {connector.region_name}")
        logger.info(f"- Selected Board: {connector.select_board}")
        
        # Test S3 connection
        try:
            connector.s3.head_bucket(Bucket=connector.bucket_name)
            logger.info("Successfully connected to S3 bucket")
        except Exception as e:
            logger.error(f"Failed to connect to S3 bucket: {str(e)}")
            raise
        
        yield from connector.stream_csv_data(latest_date_processed)
    except Exception as e:
        logger.error(f"Error in load_all_csv_data_from_s3: {str(e)}")
        raise