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

# Load environment variables first
load_dotenv()

from config.settings import Config
from knowledge_agents.data_ops import DataConfig, DataOperations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Detect environment
def detect_environment():
    """Detect the current execution environment."""
    if os.path.exists('/.dockerenv'):
        return "docker"
    elif os.environ.get('REPL_ID'):
        return "replit"
    else:
        return "local"

ENV_TYPE = detect_environment()
logger.info(f"Detected environment: {ENV_TYPE}")

def is_scheduler_running() -> bool:
    """Check if scheduler is already running."""
    data_ops = DataOperations(DataConfig.from_config())
    scheduler_marker = data_ops.config.root_data_path / ".scheduler_running"
    return scheduler_marker.exists()

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
        # Get settings from Config
        paths = Config.get_paths()
        processing_settings = Config.get_processing_settings()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()
        
        # Get data retention period from environment or config
        data_retention_days = int(os.environ.get(
            'DATA_RETENTION_DAYS', 
            processing_settings.get('retention_days', 30)
        ))
        
        logger.info(f"Data retention period: {data_retention_days} days")
        
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
        
        if is_initial_startup:
            logger.info("Initial startup detected. Following Chanscope startup approach...")
            
            # For initial setup, perform complete data refresh but skip embedding generation
            # to avoid long startup times (as per Chanscope approach)
            logger.info("Phase 1: Loading and stratifying data (force_refresh=True, skip_embeddings=True)")
            await operations.ensure_data_ready(force_refresh=True, skip_embeddings=True)
            
            # Schedule embedding generation as a second phase (as per Chanscope approach)
            logger.info("Phase 2: Generating embeddings separately")
            embedding_result = await operations.generate_embeddings()
            logger.info(f"Embedding generation completed: {embedding_result}")
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
    args = parser.parse_args()
    
    if args.run_once:
        logger.info("Running in single-execution mode (--run_once)")
        await run_update()
        return
    
    # Get update interval from environment or default to 60 minutes
    update_interval_minutes = int(os.environ.get('DATA_UPDATE_INTERVAL_MINUTES', 60))
    logger.info(f"Starting scheduled updates with interval: {update_interval_minutes} minutes")
    
    # Log environment information
    logger.info(f"Environment: {ENV_TYPE}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Python version: {sys.version}")
    
    # Create a marker file to indicate the scheduler is running
    scheduler_marker = Path("data") / ".scheduler_running"
    scheduler_marker.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(scheduler_marker, "w") as f:
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Environment: {ENV_TYPE}\n")
            f.write(f"Hostname: {socket.gethostname()}\n")
            f.write(f"PID: {os.getpid()}\n")
    except Exception as e:
        logger.warning(f"Could not create scheduler marker: {e}")
    
    while True:
        try:
            await run_update()
            logger.info(f"Waiting {update_interval_minutes} minutes until next update...")
            await asyncio.sleep(update_interval_minutes * 60)
        except Exception as e:
            logger.error(f"Error in update cycle: {str(e)}", exc_info=True)
            # Wait a bit before retrying to avoid rapid failure loops
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())