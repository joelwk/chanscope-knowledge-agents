import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator, List, Dict, Any
from datetime import datetime
import pandas as pd
import boto3
from dateutil import tz
from config.settings import Config
from config.config_utils import parse_filter_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class S3Handler:
    """Handler for S3 operations using configuration settings."""

    def __init__(self, local_data_path: str = '.'):
        aws_settings = Config.get_aws_settings()
        processing_settings = Config.get_processing_settings()
        paths = Config.get_paths()

        self.bucket_name = aws_settings['s3_bucket'].strip()
        self.bucket_prefix = aws_settings['s3_bucket_prefix'].strip()
        self.local_path = Path(paths['root_data_path']).resolve()
        self.region_name = aws_settings['aws_default_region'].strip()
        self.aws_access_key_id = aws_settings['aws_access_key_id'].strip()
        self.aws_secret_access_key = aws_settings['aws_secret_access_key'].strip()
        
        # Fix for handling None value in select_board
        select_board = processing_settings.get('select_board')
        self.select_board = select_board.strip() if select_board else None
        
        self.s3 = self._create_s3_client()

    @property
    def is_configured(self) -> bool:
        """Check if S3 is properly configured with valid credentials."""
        return bool(
            self.aws_access_key_id and
            self.aws_secret_access_key and
            self.region_name and
            self.bucket_name and
            self.bucket_prefix
        )

    def _create_s3_client(self):
        """Create and return an S3 client with credentials from Config."""
        try:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
            )
            return session.client('s3')
        except Exception as e:
            logger.exception("Failed to create S3 client")
            raise

    def get_s3_client(self):
        return self.s3

    def get_latest_data_key(self) -> Optional[str]:
        """
        Get the key of the most recently modified CSV file in S3.
        
        Returns:
            Optional[str]: The S3 key of the most recently modified CSV file, or None if no CSV files found
        """
        if not self.is_configured:
            logger.warning("S3 not configured, cannot get latest data key")
            return None
            
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.bucket_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No objects found in bucket {self.bucket_name} with prefix {self.bucket_prefix}")
                return None
                
            # Filter CSV files and apply board filter if needed
            csv_files = []
            for item in response.get('Contents', []):
                key = item['Key']
                last_modified = item['LastModified']
                
                if key.endswith('.csv'):
                    # Apply board filter if set
                    if self.select_board:
                        if self.select_board.lower() in key.lower():
                            csv_files.append((key, last_modified))
                    else:
                        csv_files.append((key, last_modified))
            
            if not csv_files:
                logger.warning(f"No CSV files found in bucket {self.bucket_name} with prefix {self.bucket_prefix}")
                if self.select_board:
                    logger.warning(f"Board filter '{self.select_board}' may be excluding all files")
                return None
                
            # Sort by last modified date (newest first)
            csv_files.sort(key=lambda x: x[1], reverse=True)
            
            # Return the most recently modified file key
            latest_key = csv_files[0][0]
            logger.info(f"Latest S3 data key: {latest_key} (modified: {csv_files[0][1]})")
            return latest_key
            
        except Exception as e:
            logger.exception(f"Error getting latest data key: {e}")
            return None
    
    def get_object_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 object.
        
        Args:
            s3_key: The S3 key of the object
            
        Returns:
            Optional[Dict[str, Any]]: Object metadata or None if object doesn't exist
        """
        if not self.is_configured:
            logger.warning("S3 not configured, cannot get object metadata")
            return None
            
        try:
            response = self.s3.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response
        except Exception as e:
            logger.exception(f"Error getting object metadata for {s3_key}: {e}")
            return None

    def file_exists_in_s3(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def _upload_file(self, local_file: str, key: str) -> None:
        """Upload a file to S3."""
        self.s3.upload_file(local_file, self.bucket_name, key)
        logger.info(f"Uploaded {local_file} to {self.bucket_name}/{key}")

    def upload_dir(self, dir_key: str) -> None:
        """Upload a local directory (recursively) to S3, skipping existing files."""
        for root, _, files in os.walk(self.local_path):
            for file in files:
                local_file = Path(root) / file
                relative_path = local_file.relative_to(self.local_path)
                s3_key = str(Path(dir_key) / relative_path)
                if not s3_key.endswith('/') and not self.file_exists_in_s3(s3_key):
                    self._upload_file(str(local_file), s3_key)
                else:
                    logger.info(f"Skipping {s3_key}; already exists in S3.")

    def download_file(self, s3_key: str, local_file_path: Optional[str] = None) -> None:
        """Download a single file from S3."""
        if not local_file_path:
            local_file_path = self.local_path / Path(s3_key).name
        local_file = Path(local_file_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket_name, s3_key, str(local_file))
        logger.info(f"Downloaded {s3_key} to {local_file}")

    def download_dir(self, dir_key: str, local_file_path: Optional[str] = None) -> None:
        """Download all files under a given S3 directory."""
        if not local_file_path:
            local_file_path = self.local_path / Path(dir_key).name
        local_dir = Path(local_file_path)
        local_dir.mkdir(parents=True, exist_ok=True)

        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=dir_key):
            for item in result.get('Contents', []):
                self.download_file(item['Key'])

    async def stream_csv_data(self, filter_date: Optional[str] = None) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Stream CSV data from S3 in chunks after filtering by date.
        Cleans columns and dates and yields processed DataFrame chunks.
        """
        try:
            chunk_settings = Config.get_chunk_settings()
            column_settings = Config.get_column_settings()
            required_columns = list(column_settings['column_types'].keys())
            chunk_size = chunk_settings['processing_chunk_size']
            logger.info("=== Starting CSV Data Stream ===")
            logger.info(f"Chunk size: {chunk_size}, Filter date: {filter_date}")

            latest_date = None
            if filter_date:
                try:
                    filter_date = parse_filter_date(filter_date)
                    latest_date = pd.to_datetime(filter_date, utc=True)
                    logger.info(f"Parsed filter date: {latest_date.isoformat()} UTC")
                except ValueError as e:
                    logger.error("Error parsing filter date", exc_info=True)
                    raise

            total_rows_processed = 0
            files_processed = 0
            csv_files = self._get_filtered_csv_files(latest_date)
            logger.info(f"Found {len(csv_files)} CSV files to process")

            for s3_key in csv_files:
                files_processed += 1
                temp_file = Path(f"temp_{Path(s3_key).name}")
                try:
                    logger.info(f"Processing file {files_processed}/{len(csv_files)}: {s3_key}")
                    self.s3.download_file(self.bucket_name, s3_key, str(temp_file))

                    chunk_iterator = pd.read_csv(
                        temp_file,
                        chunksize=chunk_size,
                        on_bad_lines='skip',
                        encoding='utf-8',
                        dtype=str,
                        skipinitialspace=True,
                        skip_blank_lines=True,
                    )

                    for chunk in chunk_iterator:
                        if chunk.empty:
                            continue

                        # Clean and validate columns
                        chunk.columns = [col.strip().replace('\r', '').replace('\n', '') for col in chunk.columns]
                        missing_cols = set(required_columns) - set(chunk.columns)
                        if missing_cols:
                            # Attempt case-insensitive matching
                            col_map = {col.lower(): col for col in chunk.columns}
                            for req in list(missing_cols):
                                if req.lower() in col_map:
                                    chunk.rename(columns={col_map[req.lower()]: req}, inplace=True)
                                    missing_cols.remove(req)
                            if missing_cols:
                                logger.error(f"Missing columns in {s3_key}: {missing_cols}")
                                continue

                        for col in chunk.columns:
                            chunk[col] = chunk[col].astype(str).str.strip()

                        # Parse date column with cleanup
                        try:
                            chunk['posted_date_time'] = pd.to_datetime(
                                chunk['posted_date_time'],
                                format='mixed',
                                utc=True,
                                errors='coerce'
                            )
                        except Exception as e:
                            logger.warning(f"Date conversion issue in {s3_key}, attempting cleanup", exc_info=True)
                            chunk['posted_date_time'] = pd.to_datetime(
                                chunk['posted_date_time'].str.strip().str.replace('\r', ''),
                                format='mixed',
                                utc=True,
                                errors='coerce'
                            )

                        valid_mask = ~chunk['posted_date_time'].isna()
                        if not valid_mask.all():
                            logger.warning(f"Removed {(~valid_mask).sum()} rows with invalid dates in {s3_key}")
                            chunk = chunk[valid_mask]

                        if not chunk.empty:
                            total_rows_processed += len(chunk)
                            logger.info(
                                f"Yielding chunk with {len(chunk)} rows from {s3_key} (Total processed: {total_rows_processed})"
                            )
                            yield chunk
                except Exception as e:
                    logger.exception(f"Error processing file {s3_key}")
                finally:
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except Exception:
                            logger.warning(f"Failed to delete temporary file {temp_file}")
            logger.info("=== Stream Complete ===")
            logger.info(f"Total files processed: {files_processed}, Total rows: {total_rows_processed}")
        except Exception as e:
            logger.exception("Critical error in stream_csv_data")
            raise

    def _get_filtered_csv_files(self, latest_date: Optional[datetime] = None) -> List[str]:
        """
        Get a list of CSV files from S3 that meet the date and board filtering criteria.
        """
        logger.info("=== Getting Filtered CSV Files ===")
        logger.info(f"Latest date filter: {latest_date}")
        logger.info(f"Bucket: {self.bucket_name}, Prefix: {self.bucket_prefix}")
        if self.select_board:
            logger.info(f"Board filtering enabled for: {self.select_board}")

        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.bucket_prefix)
            if 'Contents' not in response:
                logger.warning(f"No objects found in bucket {self.bucket_name} with prefix {self.bucket_prefix}")
                return []

            csv_keys = []
            total_files = 0
            filtered_by_date = 0
            filtered_by_board = 0

            for item in response.get('Contents', []):
                key = item['Key']
                if key.endswith('.csv'):
                    total_files += 1
                    file_date = item['LastModified'].astimezone(tz.UTC)
                    board_match = True if not self.select_board else self.select_board.lower() in key.lower()
                    if not board_match:
                        filtered_by_board += 1
                        continue

                    if latest_date is not None:
                        cutoff_date = latest_date - pd.Timedelta(days=30)
                        if file_date < cutoff_date:
                            filtered_by_date += 1
                            logger.info(f"Skipping {key} (date {file_date} < cutoff {cutoff_date})")
                            continue
                    csv_keys.append(key)
            logger.info(
                f"Total objects: {len(response['Contents'])}, CSV files: {total_files}, "
                f"Filtered by date: {filtered_by_date}, board: {filtered_by_board}, Selected: {len(csv_keys)}"
            )
            return csv_keys
        except Exception as e:
            logger.exception("Error getting filtered CSV files")
            raise


class CloudHandler:
    """Handles controlled asynchronous processing of CSV files."""

    def __init__(self, config: object):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._active_workers = 0
        self._max_workers = 3  # Limit concurrent workers
        self._worker_lock = asyncio.Lock()

    async def _get_worker(self) -> bool:
        async with self._worker_lock:
            if self._active_workers >= self._max_workers:
                return False
            self._active_workers += 1
            return True

    async def _release_worker(self) -> None:
        async with self._worker_lock:
            self._active_workers -= 1

    async def process_files(self, files: List[str], cutoff_date: Optional[datetime] = None) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Process a list of CSV files asynchronously using a limited worker pool.
        """
        try:
            filtered_files = await self._filter_files(files, cutoff_date)
            self._logger.info(f"Found {len(filtered_files)} CSV files to process")

            for idx, file in enumerate(filtered_files, 1):
                while not await self._get_worker():
                    await asyncio.sleep(1)
                try:
                    self._logger.info(f"Processing file {idx}/{len(filtered_files)}: {file}")
                    async for chunk in self._process_file(file):
                        yield chunk
                except Exception as e:
                    self._logger.exception(f"Error processing file {file}")
                finally:
                    await self._release_worker()
        except Exception as e:
            self._logger.exception("Error in process_files")
            raise

    async def _process_file(self, file: str) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Process a single CSV file in chunks with error handling.
        """
        chunk_size = self.config.processing_chunk_size
        total_processed = 0
        try:
            for chunk in pd.read_csv(file, chunksize=chunk_size):
                total_processed += len(chunk)
                self._logger.info(f"File {file}: Yielding chunk of {len(chunk)} rows (Total processed: {total_processed})")
                yield chunk
                await asyncio.sleep(0)
        except Exception as e:
            self._logger.exception(f"Error processing chunks in file {file}")
            raise


def parse_filter_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a filter date string into a timezone-aware datetime object."""
    if not date_str:
        logger.warning("No filter date provided, will process all available data")
        return None
    
    try:
        logger.info(f"Parsing filter date: {date_str}")
        dt = pd.to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        logger.info(f"Successfully parsed filter date to: {dt.isoformat()}")
        return dt
    except Exception as e:
        logger.error(f"Error parsing filter date '{date_str}': {e}", exc_info=True)
        return None


async def load_all_csv_data_from_s3(
    latest_date_processed: Optional[str] = None,
    chunk_size: int = 25000,
    board_id: Optional[str] = None
) -> AsyncGenerator[pd.DataFrame, None]:
    """
    Load and process CSV data from S3 incrementally.
    
    Args:
        latest_date_processed: Filter data to only include items after this date
        chunk_size: Size of data chunks to process at once
        board_id: Filter data to only include items from this board
    
    Yields:
        DataFrame chunks with filtered data
    """
    logger.info("=== Starting S3 Data Loading ===")
    logger.info(f"Latest date processed: {latest_date_processed}")
    logger.info(f"Board filter: {board_id or 'None'}")
    
    filter_date = parse_filter_date(latest_date_processed)
    if filter_date:
        logger.info(f"Using filter timestamp: {filter_date.isoformat()} UTC")
    else:
        logger.warning("No valid filter date provided, will process all available data")

    # Create S3Handler with custom processing settings if board_id is provided
    if board_id:
        # Get current processing settings
        processing_settings = Config.get_processing_settings()
        # Create a copy with the board_id
        custom_settings = dict(processing_settings)
        custom_settings['select_board'] = board_id
        # Override Config temporarily
        s3_handler = S3Handler()
        s3_handler.select_board = board_id
    else:
        s3_handler = S3Handler()
    
    try:
        try:
            s3_handler.s3.head_bucket(Bucket=s3_handler.bucket_name)
            logger.info("Connected to S3 bucket successfully")
        except Exception as e:
            logger.exception("Failed to connect to S3 bucket")
            raise

        async for chunk in s3_handler.stream_csv_data(latest_date_processed):
            yield chunk
    except Exception as e:
        logger.exception("Error in load_all_csv_data_from_s3")
        raise