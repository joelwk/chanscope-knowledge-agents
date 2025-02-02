from dotenv import load_dotenv
import asyncio
import logging
from pathlib import Path

# Load environment variables first
load_dotenv()

from config.settings import Config
from knowledge_agents.data_ops import DataConfig, DataOperations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    try:
        # Get settings from Config
        paths = Config.get_paths()
        processing_settings = Config.get_processing_settings()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()
        
        data_config = DataConfig(
            root_data_path=Path(paths['root_data_path']),
            stratified_data_path=Path(paths['stratified']),
            knowledge_base_path=Path(paths['knowledge_base']),
            temp_path=Path(paths['temp']),
            filter_date=processing_settings['filter_date'],
            sample_size=sample_settings['default_sample_size'],
            time_column=column_settings['time_column'],
            strata_column=column_settings['strata_column']
        )
        
        operations = DataOperations(data_config)
        
        # Check if this is initial startup
        if not Path(paths['root_data_path']).exists():
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