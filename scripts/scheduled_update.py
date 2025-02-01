import os
from dotenv import load_dotenv
import asyncio
import logging
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# Load environment variables first
load_dotenv()

from config.settings import Config
from knowledge_agents.data_processing.scheduler import DataScheduler
from knowledge_agents.data_ops import DataConfig, DataOperations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    try:
        config = Config()
        data_config = DataConfig(
            root_path=Path(config.ROOT_PATH),
            all_data_path=Path(config.ALL_DATA),
            stratified_data_path=Path(config.ALL_DATA_STRATIFIED_PATH),
            knowledge_base_path=Path(config.KNOWLEDGE_BASE),
            filter_date=None,  # Will be determined by DataStateManager
            sample_size=config.SAMPLE_SIZE,
            time_column=config.TIME_COLUMN,
            strata_column=config.STRATA_COLUMN if hasattr(config, 'STRATA_COLUMN') else None
        )
        
        operations = DataOperations(data_config)
        
        # Check if this is initial startup
        if not Path(config.ALL_DATA).exists():
            logger.info("Initial startup detected. Loading last week's data...")
            # For initial setup, we want a full refresh
            await operations.prepare_data(force_refresh=True)
        else:
            logger.info("Performing incremental update...")
            # For scheduled updates, we want incremental updates
            await operations.prepare_data(force_refresh=False)
            
    except Exception as e:
        logger.error(f"Error during data operation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())