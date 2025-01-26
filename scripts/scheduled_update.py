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
from knowledge_agents.data_ops import DataConfig

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
            filter_date=None,  # Will be set by scheduler methods
            sample_size=config.SAMPLE_SIZE,
            time_column=config.TIME_COLUMN,
            strata_column=config.STRATA_COLUMN if hasattr(config, 'STRATA_COLUMN') else None
        )
        
        scheduler = DataScheduler(data_config)
        
        # Check if this is initial startup
        if not Path(config.ALL_DATA).exists():
            logger.info("Initial startup detected. Loading last week's data...")
            await scheduler.initialize_storage()
        else:
            logger.info("Performing hourly update...")
            await scheduler.update_storage()
            
    except Exception as e:
        logger.error(f"Error during data operation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())