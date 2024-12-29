import pandas as pd
import os
import shutil
import logging
from .data_processing.sampler import Sampler
from .data_processing.cloud_handler import load_all_csv_data_from_s3
from .stratified_ops import split_dataframe
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load environment variables
load_dotenv()

# Get environment variables with defaults
CONFIGS = {
    'ROOT_PATH': os.getenv('ROOT_PATH', '/app/data'),
    'ALL_DATA': os.getenv('ALL_DATA', '/app/data/all_data.csv'),
    'ALL_DATA_STRATIFIED_PATH': os.getenv('ALL_DATA_STRATIFIED_PATH', '/app/data/stratified'),
    'KNOWLEDGE_BASE_PATH': os.getenv('KNOWLEDGE_BASE', '/app/data/knowledge_base.csv'),
    'SAMPLE_SIZE': int(os.getenv('SAMPLE_SIZE', '1000')),
    'FILTER_DATE': os.getenv('FILTER_DATE', '2024-12-24'),
}

def remove_directory_files(directory_path: str) -> None:
    """
    Removes all files in the specified directory.
    Args:
    directory_path (str): The path to the directory.
    """
    logger.info(f"\n=== Starting directory cleanup: {directory_path} ===")
    if os.path.exists(directory_path):
        logger.info(f"Directory exists. Current contents:")
        for filename in os.listdir(directory_path):
            logger.info(f"  - {filename}")
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    logger.info(f"Removing file: {file_path}")
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    logger.info(f"Removing directory: {file_path}")
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')
        logger.info(f"Cleanup completed. Directory contents after removal:")
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                logger.info(f"  - {filename}")
        else:
            logger.info("  Directory no longer exists")
    else:
        logger.info(f"Directory '{directory_path}' does not exist.")
        
def create_directory_if_not_exists(path: str) -> None:
    """
    Creates a directory if it does not already exist.
    Args:
    path (str): The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' created successfully.")
    else:
        logger.info(f"Directory '{path}' already exists.")

def load_and_save_new_data(latest_date_processed: str, output_path: str) -> None:
    """
    Loads new data from S3, saves it to a CSV file.
    Args:
    latest_date_processed (str): The latest date processed.
    output_path (str): The output path for the CSV file.
    """
    new_data = load_all_csv_data_from_s3(latest_date_processed=latest_date_processed)
    new_data.to_csv(output_path, index=False)

def stratify_data(input_path: str, time_column: str, strata_column: str, initial_sample_size: int) -> pd.DataFrame:
    """
    Loads data, stratifies it, and returns the stratified sample.
    Args:
    input_path (str): The input path to the data CSV file.
    time_column (str): The time column.
    strata_column (str): The column to stratify by.
    initial_sample_size (int): The initial sample size.
    
    Returns:
    pd.DataFrame: The stratified data sample.
    """
    all_data = pd.read_csv(input_path, low_memory=False)
    sampler = Sampler(
        filter_date=CONFIGS['FILTER_DATE'],
        time_column=time_column,
        strata_column=strata_column,
        initial_sample_size=initial_sample_size
    )
    return sampler.stratified_sample(all_data)

def prepare_data(process_new: bool = False):
    """
    Prepare data using environment variables for configuration.
    
    Args:
        process_new (bool): Whether to process new data from S3. If True, existing data will be removed.
    """
    logger.info(f"\n=== Starting data preparation (process_new={process_new}) ===")
    logger.info(f"ROOT_PATH: {CONFIGS['ROOT_PATH']}")
    logger.info(f"ALL_DATA: {CONFIGS['ALL_DATA']}")
    logger.info(f"ALL_DATA_STRATIFIED_PATH: {CONFIGS['ALL_DATA_STRATIFIED_PATH']}")
    
    # Create directory for all data if it doesn't exist
    create_directory_if_not_exists(CONFIGS['ROOT_PATH'])
    # Create directory for stratified data if it doesn't exist
    create_directory_if_not_exists(CONFIGS['ALL_DATA_STRATIFIED_PATH'])
    
    if process_new:
        logger.info("\n=== Processing new data requested ===")
        # Remove all existing files for a fresh start
        logger.info("Removing all existing files for fresh start...")
        remove_directory_files(CONFIGS['ROOT_PATH'])
        
        # Load and save new data from S3
        logger.info(f"\n=== Loading fresh data from S3 ===")
        logger.info(f"Filter date: {CONFIGS['FILTER_DATE']}")
        logger.info(f"Target path: {CONFIGS['ALL_DATA']}")
        load_and_save_new_data(CONFIGS['FILTER_DATE'], CONFIGS['ALL_DATA'])
        logger.info(f"Data loaded successfully")
    
    # Get sampling configuration from environment
    time_column = os.getenv('TIME_COLUMN', 'posted_date_time')
    strata_column = os.getenv('STRATA_COLUMN', 'None')
    if strata_column.lower() == 'none':
        strata_column = None
    
    logger.info(f"\n=== Starting stratification ===")
    logger.info(f"Time column: {time_column}")
    logger.info(f"Strata column: {strata_column}")
    logger.info(f"Sample size: {CONFIGS['SAMPLE_SIZE']}")
    
    # Stratify data
    logger.info("\nLoading data for stratification...")
    stratified_data = stratify_data(
        input_path=CONFIGS['ALL_DATA'],
        time_column=time_column,
        strata_column=strata_column,
        initial_sample_size=CONFIGS['SAMPLE_SIZE']
    )
    logger.info(f"Successfully loaded and stratified data with {len(stratified_data)} rows")
    
    logger.info("\nSplitting stratified data...")
    split_dataframe(
        stratified_data, 
        fraction=0.1, 
        stratify_column=time_column, 
        save_directory=CONFIGS['ALL_DATA_STRATIFIED_PATH'], 
        seed=42, 
        file_format='csv'
    )
    
    logger.info("\n=== Data preparation completed successfully! ===")
    return "Data preparation completed."