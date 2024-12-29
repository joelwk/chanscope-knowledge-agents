import os
import json
import boto3
import pandas as pd
import logging
from dateutil import tz
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Get environment variables with defaults
bucket_name = os.getenv('S3_BUCKET', 'rolling-data')
bucket_prefix = os.getenv('S3_BUCKET_PREFIX', 'data')

class S3Handler:
    def __init__(self, local_data_path='./s3-data', region_name='us-east-1'):
        self.bucket_name = os.getenv('S3_BUCKET', 'rolling-data')
        self.bucket_prefix = os.getenv('S3_BUCKET_PREFIX', 'data')
        self.local_path = os.path.normpath(local_data_path) + os.sep
        self.region_name = region_name
        self.s3 = self._create_s3_client()
        
    def _create_s3_client(self):
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=self.region_name
        )
        return session.client('s3')
    
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def get_s3_client(self):
        return self.s3

    def file_exists_in_s3(self, s3_key):
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3.exceptions.ClientError:
            return False
            
    def upload_dir(self, dir_key):
        for root, _, files in os.walk(self.local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, self.local_path)
                s3_key = os.path.join(dir_key, relative_path).replace(os.sep, '/')
                print(f"Bucket name: {self.bucket_name}, Type: {type(self.bucket_name)}")
                if not local_file.endswith('/') and not local_file.endswith('\\'):
                    if not self.file_exists_in_s3(s3_key):
                        self._upload_file(local_file, s3_key)
                    else:
                         print(f"Skipping {s3_key}, already exists in S3.")

    def download_dir(self, dir_key):
        if not local_file_path:
            local_file_path = os.path.join(self.local_path, os.path.basename(dir_key))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=self.bucket_name, Prefix=dir_key):
            for file in result.get('Contents', []):
                self.download_file(file['Key'])

    def download_file(self, s3_key, local_file_path=None, file_type='model', custom_objects=None):
        if not local_file_path:
            local_file_path = os.path.join(self.local_path, os.path.basename(s3_key))
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, s3_key, local_file_path)
        print(f"Downloaded file {s3_key} to {local_file_path}")
        if file_type == 'model':
            if custom_objects:
                with custom_object_scope(custom_objects):
                    return load_model(local_file_path)
            else:
                return load_model(local_file_path)
        elif file_type in ['csv', 'parquet']:
            if file_type == 'csv':
                return pd.read_csv(local_file_path)
            elif file_type == 'parquet':
                return pd.read_parquet(local_file_path)

    def _upload_file(self, local_file, key):
        self.s3.upload_file(local_file, self.bucket_name, key)
        logger.info(f"Uploaded file {local_file} to {self.bucket_name}/{key}")
        
def load_all_csv_data_from_s3(bucket=None, s3_prefix=None, select_board=None, latest_date_processed=None):
    bucket = bucket or os.getenv('S3_BUCKET', 'rolling-data')
    s3_prefix = s3_prefix or os.getenv('S3_BUCKET_PREFIX', 'data')
    
    logging.info(f"Loading all CSV data from S3 bucket: {bucket}, prefix: {s3_prefix}")
    logging.info(f"AWS Region: {os.getenv('AWS_DEFAULT_REGION', 'us-east-1')}")
    logging.info(f"Using AWS Access Key ID: {os.getenv('AWS_ACCESS_KEY_ID', 'Not Set')[:5]}...")
    
    try:
        connector = S3Handler()
        s3_client = connector.get_s3_client()
        logging.info("Successfully created S3 client")
        
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
            logging.info(f"Successfully listed objects in bucket {bucket}")
        except Exception as e:
            logging.error(f"Failed to list objects in S3 bucket: {str(e)}")
            if hasattr(e, 'response'):
                logging.error(f"Error response: {e.response}")
            raise
        
        if 'Contents' not in response:
            logging.warning(f"No files found in bucket {bucket} with prefix {s3_prefix}")
            return pd.DataFrame()
        
        if latest_date_processed:
            latest_date_processed = pd.to_datetime(latest_date_processed, utc=True)
            if latest_date_processed.tzinfo is None or latest_date_processed.tzinfo != tz.UTC:
                latest_date_processed = latest_date_processed.astimezone(tz.UTC)
            logging.info(f"Latest date processed: {latest_date_processed}")
        
        # Filter files based on board prefix and latest_date_processed
        filtered_files = []
        for item in response.get('Contents', []):
            if item['Key'].endswith('.csv'):
                if select_board is None or f'chanscope_{select_board}' in item['Key']:
                    if latest_date_processed is None or item['LastModified'].astimezone(tz.UTC) > latest_date_processed:
                        filtered_files.append(item)
                        logging.debug(f"Found matching file: {item['Key']}")
        
        csv_objects = [item['Key'] for item in filtered_files]
        logging.info(f"Found {len(csv_objects)} CSV files to process")
        
        if not csv_objects:
            logging.warning("No new CSV files to process since the last update.")
            return pd.DataFrame()
        
        all_data_frames = []
        for file_key in csv_objects:
            try:
                temp_file_path = f"temp_{file_key.replace('/', '_')}"
                logging.info(f"Downloading {file_key} to {temp_file_path}")
                s3_client.download_file(bucket, file_key, temp_file_path)
                df = pd.read_csv(temp_file_path, low_memory=False)
                logging.info(f"Successfully loaded {file_key} with {len(df)} rows")
                all_data_frames.append(df)
            except Exception as e:
                logging.error(f"Error processing file {file_key}: {str(e)}")
                if hasattr(e, 'response'):
                    logging.error(f"Error response: {e.response}")
                raise
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        combined_data = pd.concat(all_data_frames, ignore_index=True) if all_data_frames else pd.DataFrame()
        logging.info(f"Combined data contains {len(combined_data)} rows.")
        return combined_data
        
    except Exception as e:
        logging.error(f"Critical error in load_all_csv_data_from_s3: {str(e)}")
        if hasattr(e, 'response'):
            logging.error(f"Error response: {e.response}")
        raise