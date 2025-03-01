"""Data processing and inference runner module."""
import asyncio
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Tuple, List, Union, Dict, Any, Optional
import traceback
import pandas as pd
import nest_asyncio
import IPython
import time
import numpy as np

# Internal imports
from .model_ops import (
    KnowledgeAgent, 
    ModelConfig,
    ModelConfigurationError,
    ModelOperationError,
    ModelOperation,
    ModelProvider
)
from .data_ops import (
    DataConfig, 
    DataOperations
)
from .inference_ops import process_multiple_queries_efficient as process_multiple_queries
from .embedding_ops import get_agent
from api.errors import ProcessingError  # Add this import for error handling

# Configuration imports
from config.config_utils import (
    build_model_config, 
    validate_model_config,
)
from config.logging_config import get_logger
from api.cache import cache, CACHE_HITS, CACHE_MISSES, CACHE_ERRORS

# Setup logging using centralized configuration
logger = get_logger(__name__)

# Enable nested asyncio for Jupyter notebooks
try:
    nest_asyncio.apply()
except Exception:
    pass

def _get_config_hash(config: ModelConfig) -> str:
    """Generate a hash of configuration settings that affect data processing.
    
    This function extracts relevant settings that determine the data content
    and creates a unique hash to identify the configuration state.
    
    Args:
        config: ModelConfig instance containing configuration settings
        
    Returns:
        str: MD5 hash of the configuration settings
    """
    # Extract relevant settings that affect data processing
    settings = {
        'filter_date': str(config.filter_date) if config.filter_date else None,
        'sample_size': config.sample_size,
        'paths': {
            'root_data_path': str(config.paths.get('root_data_path')),
            'stratified': str(config.paths.get('stratified')),
            'temp': str(config.paths.get('temp'))
        }
    }
    
    # Create hash
    settings_str = json.dumps(settings, sort_keys=True)
    return hashlib.md5(settings_str.encode()).hexdigest()

async def _prepare_data_if_needed(
    config: ModelConfig, 
    data_ops: DataOperations,
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> pd.DataFrame:
    """Prepare data based on force_refresh parameter.
    
    For force_refresh=True:
    - Check if complete_data.csv is up-to-date with S3, only refresh if not up-to-date
    - Create new stratified sample and embeddings
    
    For force_refresh=False:
    - Only check if complete_data.csv exists, not whether it's fresh
    - Skip creating new stratified data and embeddings unless completely missing
    - Handle embedding mismatches by only generating missing embeddings
    
    Args:
        config: ModelConfig instance containing configuration settings
        data_ops: DataOperations instance for data handling
        force_refresh: Whether to force refresh stratified data and embeddings
        skip_embeddings: Whether to skip embedding generation entirely
        
    Returns:
        pd.DataFrame: Stratified data for use in inference
    """
    start_time = time.time()
    try:
        logger.info(f"Preparing data (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})")
        
        # Step 1: Check for embedding mismatch but only address it if force_refresh=False
        if not force_refresh and not skip_embeddings:
            # Load stratified data
            stratified_file = data_ops.config.stratified_data_path / 'stratified_sample.csv'
            embeddings_path = data_ops.config.stratified_data_path / 'embeddings.npz'
            thread_id_map_path = data_ops.config.stratified_data_path / 'thread_id_map.json'
            
            if (stratified_file.exists() and embeddings_path.exists() and thread_id_map_path.exists()):
                try:
                    # Load stratified data first to check counts
                    stratified_data = await data_ops._load_stratified_data()
                    stratified_count = len(stratified_data) if stratified_data is not None else 0
                    
                    if stratified_count > 0:
                        # Load embeddings to check count
                        with np.load(embeddings_path) as data:
                            embeddings = data.get('embeddings')
                        with open(thread_id_map_path, 'r') as f:
                            thread_id_map = json.load(f)
                        
                        embedding_count = len(embeddings) if embeddings is not None else 0
                        
                        # If we have a partial embedding set, only generate missing ones
                        if 0 < embedding_count < stratified_count:
                            logger.info(f"Embedding mismatch detected: {embedding_count} embeddings for {stratified_count} records")
                            
                            # Generate only the missing embeddings without full refresh
                            await data_ops.generate_missing_embeddings(stratified_data, thread_id_map)
                            logger.info("Generated missing embeddings without full data refresh")
                except Exception as e:
                    logger.warning(f"Error checking for embedding mismatch: {e}")
                    # Continue with normal processing
        
        # Step 2: Now proceed with standard data preparation based on force_refresh
        # Let DataOperations handle the data preparation with our updated logic
        await data_ops.ensure_data_ready(
            force_refresh=force_refresh,
            max_workers=config.max_workers
        )
        
        # Load stratified data for inference
        stratified_data = await data_ops._load_stratified_data()
        
        # Check data validity
        if stratified_data is None or stratified_data.empty:
            raise ProcessingError(
                message="Stratified data is empty or not available",
                operation="load_stratified_data"
            )
        
        data_rows = len(stratified_data)
        logger.info(f"Data preparation complete: {data_rows} rows available " +
                   f"(took {round((time.time() - start_time) * 1000, 2)}ms)")
        
        # Return data for use in inference
        return stratified_data
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {str(e)}")
        raise ProcessingError(
            message=f"Required data file not found: {str(e)}",
            operation="data_preparation",
            resource=str(e.filename) if hasattr(e, 'filename') else None,
            original_error=e
        )

async def _run_inference_async(
    query: str,
    data_config: ModelConfig,
    agent: KnowledgeAgent,
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Run inference operations asynchronously.
    
    Args:
        query: The query to process
        data_config: Configuration for data processing
        agent: KnowledgeAgent instance to use for processing
        force_refresh: Whether to force refresh the data
        skip_embeddings: Whether to skip embedding generation
        
    Returns:
        Tuple[List[Dict[str, Any]], str]: Chunks and summary
    """
    try:
        # Model settings are already validated in build_unified_config
        # Prepare data only if needed
        data_ops = DataOperations(DataConfig.from_config())
        stratified_data = await _prepare_data_if_needed(
            data_config,
            data_ops,
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
        
        # Get providers from config
        providers = {
            ModelOperation.EMBEDDING: data_config.get_provider(ModelOperation.EMBEDDING),
            ModelOperation.CHUNK_GENERATION: data_config.get_provider(ModelOperation.CHUNK_GENERATION),
            ModelOperation.SUMMARIZATION: data_config.get_provider(ModelOperation.SUMMARIZATION)
        }
        
        # Process single query using batch processing function
        results = await process_multiple_queries(
            queries=[query],  # Wrap single query in list
            agent=agent,
            stratified_data=stratified_data,  # Pass stratified data directly
            chunk_batch_size=data_config.get_batch_size(ModelOperation.CHUNK_GENERATION),
            summary_batch_size=data_config.get_batch_size(ModelOperation.SUMMARIZATION),
            max_workers=data_config.max_workers,
            providers=providers
        )
        
        # Unpack results for single query case
        chunks, summary = results[0]
        
        return chunks, summary
        
    except ModelConfigurationError as e:
        logger.error(f"Configuration error in inference processing: {str(e)}")
        raise
    except ModelOperationError as e:
        logger.error(f"Operation error in inference processing: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in inference processing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def run_inference(
    query: str,
    config: ModelConfig,
    agent: Optional[KnowledgeAgent] = None,
    force_refresh: bool = False,
    skip_embeddings: bool = False
) -> Union[Tuple[List[Dict[str, Any]], str], "asyncio.Future"]:
    """
    Process a query using the inference pipeline.
    
    Args:
        query: The query to process
        config: ModelConfig instance containing all configuration settings
        agent: Optional KnowledgeAgent instance (will be created if not provided)
        force_refresh: Whether to force refresh the data
        skip_embeddings: Whether to skip embedding generation
        
    Returns:
        Tuple containing relevant chunks and summary text, or Future if running in async context
    """
    # Validate configuration before processing
    validate_model_config(config)
    
    async def _run():
        nonlocal agent
        if agent is None:
            agent = await get_agent()
        
        return await _run_inference_async(
            query=query,
            data_config=config,
            agent=agent,
            force_refresh=force_refresh,
            skip_embeddings=skip_embeddings
        )
    
    # If we're already in an event loop, return a coroutine
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return _run()
    except RuntimeError:
        pass
    
    # Otherwise, create a new event loop and run synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()

def main():
    """Main entry point for running inference from command line."""
    if len(sys.argv) < 2:
        print("Usage: python -m knowledge_agents.run <query>")
        sys.exit(1)
        
    query = sys.argv[1]
    config = build_model_config()
    
    try:
        chunks, summary = run_inference(query, config)
        print("\nRelevant Chunks:")
        for chunk in chunks:
            print(f"\n- {chunk['text']}")
        print(f"\nSummary: {summary}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()