from dotenv import load_dotenv
import asyncio
import logging
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import sys
import traceback
import socket
import shutil

# Load environment variables first
load_dotenv()

from config.settings import Config
from knowledge_agents.data_ops import DataConfig, DataOperations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Detect environment
def detect_environment():
    """Detect the current execution environment."""
    if os.path.exists('/app'):
        return "docker"
    elif os.environ.get('REPL_ID'):
        return "replit"
    else:
        return "local"

ENV_TYPE = detect_environment()
logger.info(f"Detected environment: {ENV_TYPE}")

# Get environment-specific paths
def get_environment_paths():
    """Get environment-specific paths based on detected environment."""
    if ENV_TYPE == "replit":
        # Replit-specific paths
        repl_home = os.environ.get('REPL_HOME', os.getcwd())
        return {
            'root_data_path': f"{repl_home}/data",
            'stratified': f"{repl_home}/data/stratified",
            'temp': f"{repl_home}/temp_files",
            'logs': f"{repl_home}/logs",
            'test_results': f"{repl_home}/test_results",
        }
    else:
        # Default paths from config
        return Config.get_paths()

def setup_environment():
    """Set up environment-specific directories and configurations."""
    paths = get_environment_paths()
    
    # Create necessary directories
    for path_key, path_value in paths.items():
        # Convert PosixPath to string before checking endswith
        path_str = str(path_value)
        if path_key != 'temp' and not path_str.endswith('.json'):  # Skip temp files and JSON configs
            os.makedirs(path_value, exist_ok=True)
            
    # Additional Replit-specific setup
    if ENV_TYPE == "replit":
        # Create additional directories needed in Replit
        os.makedirs(f"{paths['root_data_path']}/shared", exist_ok=True)
        os.makedirs(f"{paths['root_data_path']}/logs", exist_ok=True)
        os.makedirs(f"{paths['root_data_path']}/mock", exist_ok=True)
        
        # Set Replit-specific environment variables
        if not os.environ.get('EMBEDDING_BATCH_SIZE'):
            os.environ['EMBEDDING_BATCH_SIZE'] = '5'  # Smaller batch size for Replit's limited resources
        if not os.environ.get('CHUNK_BATCH_SIZE'):
            os.environ['CHUNK_BATCH_SIZE'] = '5'
        if not os.environ.get('PROCESSING_CHUNK_SIZE'):
            os.environ['PROCESSING_CHUNK_SIZE'] = '1000'
            
        # Check if mock data should be used
        use_mock_data = os.environ.get('USE_MOCK_DATA', 'false').lower() in ('true', '1', 'yes')
        logger.info(f"USE_MOCK_DATA is set to: {os.environ.get('USE_MOCK_DATA', 'not set')} (evaluated as: {use_mock_data})")
        
        # Create sample test data for Replit if it doesn't exist and mock data is enabled
        if use_mock_data:
            mock_data_path = f"{paths['root_data_path']}/mock/sample_data.csv"
            if not os.path.exists(mock_data_path):
                logger.info("Creating sample test data for Replit environment")
                with open(mock_data_path, 'w') as f:
                    f.write("thread_id,posted_date_time,text_clean,posted_comment\n")
                    f.write("1001,2025-01-01 12:00:00,This is a test post for embedding generation,Original comment 1\n")
                    f.write("1002,2025-01-01 12:05:00,Another test post with different content,Original comment 2\n")
                    f.write("1003,2025-01-01 12:10:00,Third test post for validation purposes,Original comment 3\n")
                    f.write("1004,2025-01-01 12:15:00,Fourth test post with unique content,Original comment 4\n")
                    f.write("1005,2025-01-01 12:20:00,Fifth test post for comprehensive testing,Original comment 5\n")

                # Copy mock data to main data directory for tests if complete_data.csv doesn't exist
                complete_data_path = f"{paths['root_data_path']}/complete_data.csv"
                if not os.path.exists(complete_data_path):
                    shutil.copy(mock_data_path, complete_data_path)
                    logger.info("Copied sample test data to complete_data.csv")
        else:
            logger.info("Mock data is disabled, skipping mock data creation")

def is_scheduler_running() -> bool:
    """Check if scheduler is already running."""
    paths = get_environment_paths()
    scheduler_marker = Path(paths['root_data_path']) / ".scheduler_running"
    return scheduler_marker.exists()

async def progress_logger(label: str, start_time: float, timeout: float):
    """Logs progress every 5 seconds until the operation completes."""
    process_id = os.getpid()
    logger.info(f"Starting progress tracking for '{label}' (PID: {process_id})")
    while True:
        elapsed = asyncio.get_running_loop().time() - start_time
        progress = min(100, int((elapsed / timeout) * 100))
        remaining = max(0, timeout - elapsed)
        logger.info(f"{label} progress: {progress}% ({elapsed:.1f} sec elapsed, ~{remaining:.1f} sec remaining) [PID: {process_id}]")
        
        # Update the marker file with current progress
        try:
            paths = get_environment_paths()
            marker_file = Path(paths['root_data_path']) / ".update_in_progress"
            if marker_file.exists():
                with open(marker_file, "a") as f:
                    f.write(f"Progress [{datetime.now().isoformat()}]: {label} - {progress}%\n")
        except Exception:
            pass  # Silently ignore errors updating the marker file
            
        await asyncio.sleep(5)

async def run_with_progress(coro, label: str, timeout: float):
    """Runs a coroutine with a timeout and logs progress periodically."""
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    progress_task = asyncio.create_task(progress_logger(label, start_time, timeout))
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        progress_task.cancel()
        return result
    except asyncio.TimeoutError:
        progress_task.cancel()
        raise

async def run_update():
    """
    Run data update following the Chanscope approach:
    
    1. On initial startup:
       - Load data from S3 since DATA_RETENTION_DAYS ago
       - Stratify the data
       - Generate embeddings as a separate step
       
    2. On scheduled updates:
       - Check for new data
       - Only update if new data exists
       - Follow force_refresh=False behavior
    """
    try:
        # Get environment-specific paths
        paths = get_environment_paths()
        
        # Get other settings from Config
        processing_settings = Config.get_processing_settings()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()
        
        # Get data retention period from environment or config
        data_retention_days = int(os.environ.get(
            'DATA_RETENTION_DAYS', 
            processing_settings.get('retention_days', 30)
        ))
        
        logger.info(f"Data update process started (PID: {os.getpid()}) - Data retention period: {data_retention_days} days")
        
        # Calculate filter date if not set
        filter_date = processing_settings.get('filter_date')
        if not filter_date:
            # Default to data_retention_days ago if not specified
            retention_date = datetime.now(pytz.UTC) - timedelta(days=data_retention_days)
            filter_date = retention_date.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"No filter date specified, using retention-based date: {filter_date}")
        
        # Create data configuration
        data_config = DataConfig(
            root_data_path=Path(paths['root_data_path']),
            stratified_data_path=Path(paths['stratified']),
            temp_path=Path(paths['temp']),
            filter_date=filter_date,
            sample_size=sample_settings['default_sample_size'],
            time_column=column_settings['time_column'],
            strata_column=column_settings['strata_column']
        )
        
        # Initialize data operations
        operations = DataOperations(data_config)
        
        # Step 1: Check if application is in initial startup mode
        is_initial_startup = not Path(paths['root_data_path']).exists() or not (Path(paths['root_data_path']) / "complete_data.csv").exists()
        
        # Special handling for Replit in test mode
        use_mock_data = os.environ.get('USE_MOCK_DATA', 'false').lower() in ('true', '1', 'yes')
        logger.info(f"USE_MOCK_DATA is set to: {os.environ.get('USE_MOCK_DATA', 'not set')} (evaluated as: {use_mock_data})")
        
        if ENV_TYPE == "replit" and use_mock_data and is_initial_startup:
            logger.info("Using mock data for Replit test environment")
            # This would be handled by the setup_environment function which creates sample data
            # Just need to ensure the data is properly stratified and embeddings are generated
        elif ENV_TYPE == "replit" and not use_mock_data:
            logger.info("Mock data is disabled in Replit environment, using real data from S3")
        
        if is_initial_startup:
            logger.info("Initial startup detected. Following Chanscope startup approach...")
            
            # For initial setup, perform complete data refresh but skip embedding generation
            # to avoid long startup times (as per Chanscope approach)
            logger.info("Phase 1: Loading and stratifying data (force_refresh=True, skip_embeddings=True) with progress tracking")
            try:
                await run_with_progress(operations.ensure_data_ready(force_refresh=True, skip_embeddings=True), "Data loading", 300)
            except asyncio.TimeoutError:
                logger.error("Timeout occurred during initial data loading (ensure_data_ready did not complete within 300 seconds)")
            
            # Schedule embedding generation as a second phase (as per Chanscope approach)
            logger.info("Phase 2: Generating embeddings separately with progress tracking")
            try:
                embedding_result = await run_with_progress(operations.generate_embeddings(), "Embedding generation", 300)
                logger.info(f"Embedding generation completed: {embedding_result}")
            except asyncio.TimeoutError:
                logger.error("Timeout occurred during embedding generation (did not complete within 300 seconds)")
        else:
            logger.info("Performing incremental update following Chanscope approach...")
            # For scheduled updates, do incremental updates with force_refresh=False
            # This follows the Chanscope approach for non-forced updates
            result = await operations.ensure_data_ready(force_refresh=False)
            logger.info(f"Incremental update completed: {result}")
            
    except Exception as e:
        logger.error(f"Error during scheduled data operation: {str(e)}", exc_info=True)
        # Log additional diagnostic information
        logger.error(f"Environment: {ENV_TYPE}")
        logger.error(f"Hostname: {socket.gethostname()}")
        logger.error(f"Python version: {sys.version}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def main():
    parser = argparse.ArgumentParser(description='Run scheduled data updates')
    parser.add_argument('--run_once', action='store_true', help='Run once and exit (for testing)')
    parser.add_argument('--force_refresh', action='store_true', help='Force refresh of data and embeddings')
    parser.add_argument('--env', choices=['docker', 'replit', 'local'], help='Override environment detection')
    args = parser.parse_args()
    
    # Override environment detection if specified
    global ENV_TYPE
    if args.env:
        ENV_TYPE = args.env
        logger.info(f"Environment manually set to: {ENV_TYPE}")
    
    # Set up environment-specific configuration
    setup_environment()
    
    # Log environment and process information for all modes
    process_id = os.getpid()
    logger.info(f"Process started with PID: {process_id}")
    logger.info(f"Environment: {ENV_TYPE}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Python version: {sys.version}")
    
    # Create paths for marker files
    paths = get_environment_paths()
    
    # Create a marker file for the current execution
    execution_marker = Path(paths['root_data_path']) / ".update_in_progress"
    execution_marker.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(execution_marker, "w") as f:
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Environment: {ENV_TYPE}\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"PID: {process_id}\n")
            f.write(f"Mode: {'single-execution' if args.run_once else 'scheduler'}\n")
        logger.info(f"Created execution marker file: {execution_marker}")
    except Exception as e:
        logger.warning(f"Could not create execution marker: {e}")
    
    if args.run_once:
        logger.info("Running in single-execution mode (--run_once)")
        try:
            await run_update()
            # Remove the marker file when done
            if execution_marker.exists():
                execution_marker.unlink()
        except Exception as e:
            logger.error(f"Error in single-execution mode: {str(e)}", exc_info=True)
            # Update marker file with error information
            try:
                with open(execution_marker, "a") as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Completed with error: {datetime.now().isoformat()}\n")
            except Exception:
                pass
        return
    
    # Get update interval from environment or default to 60 minutes
    update_interval_seconds = int(os.environ.get('DATA_UPDATE_INTERVAL', 3600))
    logger.info(f"Starting scheduled updates with interval: {update_interval_seconds} seconds")
    
    # Create a marker file to indicate the scheduler is running
    scheduler_marker = Path(paths['root_data_path']) / ".scheduler_running"
    
    try:
        with open(scheduler_marker, "w") as f:
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Environment: {ENV_TYPE}\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"PID: {process_id}\n")
    except Exception as e:
        logger.warning(f"Could not create scheduler marker: {e}")
    
    while True:
        try:
            await run_update()
            logger.info(f"Waiting {update_interval_seconds} seconds until next update...")
            await asyncio.sleep(update_interval_seconds)
        except Exception as e:
            logger.error(f"Error in update cycle: {str(e)}", exc_info=True)
            # Wait a bit before retrying to avoid rapid failure loops
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())