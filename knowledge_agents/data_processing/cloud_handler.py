import boto3
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from dateutil import tz
from config.settings import Config

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

def load_all_csv_data_from_s3(latest_date_processed: str = None) -> pd.DataFrame:
    """Load all CSV data from S3 bucket with memory-efficient chunked processing."""
    connector = S3Handler()
    s3_client = connector.get_s3_client()
    chunk_size = 50000  # Process 50k rows at a time
    
    try:
        logger.info("Loading data from S3 bucket: %s", Config.S3_BUCKET)
        logger.info("Using S3 prefix: %s", Config.S3_BUCKET_PREFIX)
        logger.info("AWS Region: %s", Config.AWS_DEFAULT_REGION)
        
        # Log AWS credentials state (safely)
        if not Config.AWS_ACCESS_KEY_ID or not Config.AWS_SECRET_ACCESS_KEY:
            logger.error("AWS credentials not properly configured")
            raise ValueError("AWS credentials not properly configured")
            
        response = s3_client.list_objects_v2(Bucket=Config.S3_BUCKET, Prefix=Config.S3_BUCKET_PREFIX)
        if 'Contents' not in response:
            logger.warning(f"No files found in bucket {Config.S3_BUCKET} with prefix {Config.S3_BUCKET_PREFIX}")
            return pd.DataFrame()
        
        # Handle date filtering
        if latest_date_processed:
            latest_date_processed = pd.to_datetime(latest_date_processed, utc=True)
            if latest_date_processed.tzinfo is None or latest_date_processed.tzinfo != tz.UTC:
                latest_date_processed = latest_date_processed.astimezone(tz.UTC)
            logger.info(f"Latest date processed: {latest_date_processed}")
        
        # Filter files based on board prefix and latest_date_processed
        filtered_files = []
        for item in response.get('Contents', []):
            if item['Key'].endswith('.csv'):
                if Config.SELECT_BOARD is None or f'chanscope_{Config.SELECT_BOARD}' in item['Key']:
                    if latest_date_processed is None or item['LastModified'].astimezone(tz.UTC) > latest_date_processed:
                        filtered_files.append(item)
                        logger.debug(f"Found matching file: {item['Key']}")
        
        csv_objects = [item['Key'] for item in filtered_files]
        logger.info(f"Found {len(csv_objects)} CSV files to process")
        
        if not csv_objects:
            logger.warning("No CSV files found to process.")
            return pd.DataFrame()
        
        required_columns = {'thread_id', 'posted_date_time', 'text_clean'}
        result_df = None
        
        for file_key in csv_objects:
            try:
                temp_file = Path(f"temp_{Path(file_key).name}")
                logger.info(f"Downloading {file_key} to {temp_file}")
                s3_client.download_file(Config.S3_BUCKET, file_key, str(temp_file))
                
                # Process file in chunks
                chunk_list = []
                for chunk in pd.read_csv(temp_file, chunksize=chunk_size, low_memory=False):
                    if chunk.empty:
                        continue
                        
                    # Check required columns
                    missing_cols = required_columns - set(chunk.columns)
                    if missing_cols:
                        logger.error(f"Missing required columns in {file_key}: {missing_cols}")
                        continue
                    
                    # Validate posted_date_time format
                    try:
                        chunk['posted_date_time'] = pd.to_datetime(chunk['posted_date_time'], utc=True)
                    except Exception as e:
                        logger.error(f"Invalid datetime format in {file_key}: {str(e)}")
                        continue
                    
                    # Remove rows with NaN in required columns
                    chunk = chunk.dropna(subset=list(required_columns))
                    
                    if len(chunk) > 0:
                        chunk_list.append(chunk)
                    
                    # Free memory
                    del chunk
                    import gc
                    gc.collect()
                
                if chunk_list:
                    file_df = pd.concat(chunk_list, ignore_index=True)
                    logger.info(f"Successfully loaded {file_key} with {len(file_df)} valid rows")
                    
                    if result_df is None:
                        result_df = file_df
                    else:
                        result_df = pd.concat([result_df, file_df], ignore_index=True)
                    
                    # Free memory
                    del file_df
                    del chunk_list
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error processing file {file_key}: {str(e)}")
                continue
            finally:
                if temp_file.exists():
                    temp_file.unlink()
        
        if result_df is None or result_df.empty:
            logger.error("No valid data to return")
            return pd.DataFrame()
            
        logger.info(f"Combined data contains {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Critical error in load_all_csv_data_from_s3: {str(e)}")
        raise