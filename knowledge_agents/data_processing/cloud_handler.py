import boto3
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
        self.s3 = self._create_s3_client()
        
    def _create_s3_client(self):
        """Create an S3 client with credentials from Config."""
        session = boto3.Session(
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
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
    """Load all CSV data from S3 bucket."""
    connector = S3Handler()
    s3_client = connector.get_s3_client()
    try:
        logger.info("Loading data from S3 bucket: %s", Config.S3_BUCKET)
        logger.info("Using S3 prefix: %s", Config.S3_BUCKET_PREFIX)
        logger.info("AWS Region: %s", Config.AWS_DEFAULT_REGION)
        
        # Log AWS credentials state (safely)
        if Config.AWS_ACCESS_KEY_ID:
            logger.info("Using AWS Access Key ID: %s...", Config.AWS_ACCESS_KEY_ID[:7])
            logger.info("AWS Access Key length: %d", len(Config.AWS_ACCESS_KEY_ID))
        else:
            logger.error("AWS Access Key ID is not set")
            raise ValueError("AWS Access Key ID is not set")
            
        if not Config.AWS_SECRET_ACCESS_KEY:
            logger.error("AWS Secret Access Key is not set")
            raise ValueError("AWS Secret Access Key is not set")
        
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
                # Apply board filter if specified
                if Config.SELECT_BOARD is None or f'chanscope_{Config.SELECT_BOARD}' in item['Key']:
                    # Apply date filter if specified
                    if latest_date_processed is None or item['LastModified'].astimezone(tz.UTC) > latest_date_processed:
                        filtered_files.append(item)
                        logger.debug(f"Found matching file: {item['Key']}")
        
        csv_objects = [item['Key'] for item in filtered_files]
        logger.info(f"Found {len(csv_objects)} CSV files to process")
        
        if not csv_objects:
            logger.warning("No CSV files found to process.")
            return pd.DataFrame()
        
        all_data_frames = []
        required_columns = {'thread_id', 'posted_date_time', 'text_clean'}
        
        for file_key in csv_objects:
            try:
                temp_file = Path(f"temp_{Path(file_key).name}")
                logger.info(f"Downloading {file_key} to {temp_file}")
                s3_client.download_file(Config.S3_BUCKET, file_key, str(temp_file))
                
                # Read CSV with validation
                df = pd.read_csv(temp_file, low_memory=False)
                
                # Validate DataFrame structure
                if df.empty:
                    logger.warning(f"Empty DataFrame in file {file_key}")
                    continue
                    
                # Check required columns
                missing_cols = required_columns - set(df.columns)
                if missing_cols:
                    logger.error(f"Missing required columns in {file_key}: {missing_cols}")
                    continue
                
                # Validate posted_date_time format
                try:
                    df['posted_date_time'] = pd.to_datetime(df['posted_date_time'], utc=True)
                except Exception as e:
                    logger.error(f"Invalid datetime format in {file_key}: {str(e)}")
                    continue
                
                # Validate text_clean column
                if df['text_clean'].isna().all():
                    logger.warning(f"All text_clean values are NaN in {file_key}")
                    continue
                
                # Remove rows with NaN in required columns
                initial_rows = len(df)
                df = df.dropna(subset=list(required_columns))
                if len(df) < initial_rows:
                    logger.warning(f"Dropped {initial_rows - len(df)} rows with NaN values in {file_key}")
                
                if len(df) > 0:  # Only append if we have valid rows
                    logger.info(f"Successfully loaded {file_key} with {len(df)} valid rows")
                    all_data_frames.append(df)
                else:
                    logger.warning(f"No valid rows in {file_key} after validation")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_key}: {str(e)}")
                continue
            finally:
                if temp_file.exists():
                    temp_file.unlink()
        
        if not all_data_frames:
            logger.error("No valid DataFrames to combine")
            return pd.DataFrame()
            
        combined_data = pd.concat(all_data_frames, ignore_index=True)
        if combined_data.empty:
            logger.error("Combined DataFrame is empty")
            return pd.DataFrame()
            
        logger.info(f"Combined data contains {len(combined_data)} rows")
        return combined_data
        
    except Exception as e:
        logger.error(f"Critical error in load_all_csv_data_from_s3: {str(e)}")
        raise