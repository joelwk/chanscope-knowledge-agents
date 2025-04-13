"""
Unified ChanScope data management implementation.

This module provides a unified implementation of the Chanscope approach
that can work with either file-based storage (Docker) or database storage (Replit),
using the appropriate storage interfaces based on the detected environment.
"""

import asyncio
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Awaitable
from pathlib import Path
from datetime import datetime, timedelta, timezone
import traceback

from config.storage import StorageFactory, CompleteDataStorage, StratifiedSampleStorage, EmbeddingStorage, StateManager
from config.env_loader import detect_environment
from config.settings import Config
from knowledge_agents.data_processing.sampler import Sampler
from knowledge_agents.embedding_ops import get_relevant_content, process_batch
from knowledge_agents.data_processing.cloud_handler import S3Handler, load_all_csv_data_from_s3

# Configure logging
logger = logging.getLogger(__name__)

class ChanScopeDataManager:
    """
    Unified implementation of the Chanscope data management approach.
    
    This class uses storage interfaces to implement the Chanscope approach 
    for managing data, stratified samples, and embeddings, regardless of the 
    underlying storage implementation.
    """
    
    def __init__(
        self, 
        config,
        complete_data_storage: Optional[CompleteDataStorage] = None,
        stratified_storage: Optional[StratifiedSampleStorage] = None,
        embedding_storage: Optional[EmbeddingStorage] = None,
        state_manager: Optional[StateManager] = None
    ):
        """
        Initialize the data manager with configuration and storage implementations.
        
        Args:
            config: Configuration object with necessary paths and settings
            complete_data_storage: Implementation of CompleteDataStorage
            stratified_storage: Implementation of StratifiedSampleStorage
            embedding_storage: Implementation of EmbeddingStorage
            state_manager: Implementation of StateManager
        """
        self.config = config
        logger.info(f"Initializing ChanScopeDataManager with config: {config}")
        
        # Use provided storage implementations or create them using the factory
        if all([complete_data_storage, stratified_storage, embedding_storage, state_manager]):
            self.complete_data_storage = complete_data_storage
            self.stratified_storage = stratified_storage
            self.embedding_storage = embedding_storage
            self.state_manager = state_manager
            logger.info("Using provided storage implementations")
        else:
            # Use factory to create appropriate storage implementations for the environment
            storage = StorageFactory.create(config, getattr(config, 'env', None))
            self.complete_data_storage = complete_data_storage or storage['complete_data']
            self.stratified_storage = stratified_storage or storage['stratified_sample']
            self.embedding_storage = embedding_storage or storage['embeddings']
            self.state_manager = state_manager or storage['state']
            logger.info(f"Created storage implementations for environment: {config.env}")
        
        # Initialize other components
        self.sample_size = getattr(config, 'sample_size', 100000)
        self.sampler = Sampler(
            time_column=getattr(config, 'time_column', 'posted_date_time'),
            filter_date=getattr(config, 'filter_date', None),
            initial_sample_size=self.sample_size
        )
        
        logger.info("ChanScopeDataManager initialization complete")
    
    @classmethod
    def create_for_environment(cls, config):
        """
        Factory method to create a ChanScopeDataManager with the appropriate storage implementations.
        
        Args:
            config: Configuration object with necessary paths and settings
            
        Returns:
            ChanScopeDataManager: Instance with appropriate storage implementations for the environment
        """
        # Create storage implementations using the factory
        storage = StorageFactory.create(config, getattr(config, 'env', None))
        
        return cls(
            config=config,
            complete_data_storage=storage['complete_data'],
            stratified_storage=storage['stratified_sample'],
            embedding_storage=storage['embeddings'],
            state_manager=storage['state']
        )
    
    async def ensure_data_ready(self, force_refresh: bool = False, skip_embeddings: bool = False) -> bool:
        """
        Ensure data is ready for use, following the Chanscope approach.
        
        This implements the core Chanscope approach logic:
        - When force_refresh=True: Update complete data if needed, always create new stratified sample,
          always generate new embeddings (unless skipped)
        - When force_refresh=False: Use existing data if available, only update what's missing
        
        Args:
            force_refresh: Whether to force refresh all data
            skip_embeddings: Whether to skip embedding generation
            
        Returns:
            True if data is ready, False otherwise
        """
        try:
            # Mark operation start
            await self.state_manager.mark_operation_start("ensure_data_ready")
            logger.info(f"Ensuring data is ready (force_refresh={force_refresh}, skip_embeddings={skip_embeddings})")
            
            # Check if complete data exists and is fresh
            row_count = await self.complete_data_storage.get_row_count()
            logger.info(f"Current complete data row count: {row_count}")
            
            data_is_fresh = await self.complete_data_storage.is_data_fresh()
            logger.info(f"Complete data freshness: {data_is_fresh}")
            
            # Case A: force_refresh=True - Follow Chanscope approach for forced refresh
            if force_refresh:
                logger.info("Force refresh enabled, updating stratified data and embeddings.")
                
                # Check if data needs to be updated
                data_needs_update = row_count == 0 or not data_is_fresh
                
                if data_needs_update:
                    logger.info("Complete data needs to be updated")
                    
                    # Load data from S3
                    if row_count == 0:
                        logger.info("Database is empty, loading initial data from S3")
                        await self._load_data_from_s3()
                    else:
                        logger.info("Database exists but needs refresh, updating from S3")
                        await self._load_data_from_s3()
                else:
                    logger.info("Complete data is current, no update needed")
                
                # Always create a new stratified sample with force_refresh=True
                logger.info("Creating new stratified sample (force_refresh=True)")
                
                # Get complete data
                df = await self.complete_data_storage.get_data(self.config.filter_date)
                
                if df is not None and not df.empty:
                    # Perform stratification
                    stratified_df = self.sampler.stratified_sample(df)
                    
                    # Store the stratified sample
                    await self.stratified_storage.store_sample(stratified_df)
                    logger.info(f"Successfully created and stored stratified sample with {len(stratified_df)} rows")
                    
                    # Generate embeddings if not skipped
                    if not skip_embeddings:
                        logger.info("Initiating embedding generation (force_refresh=True)")
                        embedding_success = await self.generate_embeddings(force_refresh=True)
                        if embedding_success:
                            logger.info("Embedding generation completed successfully")
                        else:
                            logger.error("Embedding generation failed during force refresh")
                            return False
                    else:
                        logger.info("Skipping embedding generation as requested")
                else:
                    logger.warning("Complete data is empty, cannot create stratified sample")
                    return False
                
            # Case B: force_refresh=False - Only check if data exists
            else:
                # Check if data is missing and needs to be loaded
                if row_count == 0:
                    logger.info("Database is empty, loading initial data from S3")
                    await self._load_data_from_s3()
                    
                    # Get updated row count after loading data
                    row_count = await self.complete_data_storage.get_row_count()
                    logger.info(f"Updated row count after loading data: {row_count}")
                # Check if data exists but is not fresh and needs to be updated
                elif not data_is_fresh:
                    logger.info("Database exists but data is not fresh, updating from S3")
                    await self._load_data_from_s3()
                    
                    # Get updated row count after loading data
                    row_count = await self.complete_data_storage.get_row_count()
                    logger.info(f"Updated row count after refreshing data: {row_count}")
                else:
                    logger.info("Complete data exists and is fresh, no update needed")
                
                # Check if stratified sample exists
                stratified_sample = await self.stratified_storage.get_sample()
                
                if stratified_sample is None:
                    logger.info("No stratified sample found, creating one (force_refresh=False)")
                    # Get complete data
                    df = await self.complete_data_storage.get_data(self.config.filter_date)
                    
                    if df is not None and not df.empty:
                        # Perform stratification
                        stratified_df = self.sampler.stratified_sample(df)
                        
                        # Store the stratified sample
                        await self.stratified_storage.store_sample(stratified_df)
                        logger.info(f"Successfully created and stored stratified sample with {len(stratified_df)} rows")
                    else:
                        logger.warning("Complete data is empty, cannot create stratified sample")
                        return False
                else:
                    logger.info(f"Using existing stratified sample with {len(stratified_sample)} rows (force_refresh=False)")
                
                # Check if embeddings exist and generate if needed
                if not skip_embeddings:
                    embeddings_exist = await self.embedding_storage.embeddings_exist()
                    
                    if not embeddings_exist:
                        logger.info("No embeddings found, generating new ones (force_refresh=False)")
                        embedding_success = await self.generate_embeddings(force_refresh=False)
                        if embedding_success:
                            logger.info("Embedding generation completed successfully")
                        else:
                            logger.error("Embedding generation failed during non-forced refresh")
                            return False
                    else:
                        logger.info("Using existing embeddings (force_refresh=False)")
                else:
                    logger.info("Skipping embedding verification as requested")
            
            # Mark operation complete
            await self.state_manager.mark_operation_complete("ensure_data_ready", "success")
            logger.info("Data is now ready for use")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring data is ready: {e}")
            logger.error(traceback.format_exc())
            
            # Mark operation as failed
            await self.state_manager.update_state({
                "status": "error",
                "operation": "ensure_data_ready",
                "error": str(e)
            })
            
            return False
    
    async def _load_data_from_s3(self) -> bool:
        """
        Load data from S3 into the complete data storage.
        
        This method handles the S3 data loading process based on environment configuration.
        It loads all CSV files within the retention period, matching the behavior
        of the query endpoint in routes.py.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        logger.info("Loading data from S3")
        
        try:
            # Calculate retention period from current date
            current_time = pd.Timestamp.now(tz='UTC')
            retention_settings = Config.get_retention_settings()
            retention_days = retention_settings.get('retention_days', 30)  # Default to 30 days if not specified
            
            # Calculate start_time as current_time minus retention_days
            start_time = current_time - pd.Timedelta(days=retention_days)
            
            logger.info(f"Using retention period of {retention_days} days")
            logger.info(f"Data range: from {start_time.isoformat()} to {current_time.isoformat()}")
            
            # Check S3 connectivity
            s3_handler = S3Handler()
            if not s3_handler.is_configured:
                logger.error("S3 is not properly configured")
                return False
            
            # Initialize counters
            record_count = 0
            chunks_processed = 0
            
            # Get list of relevant files within retention period
            csv_files = s3_handler._get_filtered_csv_files(latest_date=start_time)
            if not csv_files:
                logger.error("No CSV files found in S3 for the specified date range")
                return False
            
            logger.info(f"Starting stream processing of {len(csv_files)} CSV files from S3")
            
            # Create a single generator for all files at once instead of per-file generators
            file_data_generator = s3_handler.stream_csv_data(start_time.isoformat())
            
            async for chunk in file_data_generator:
                chunks_processed += 1
                
                # Process date column
                time_column = getattr(self.config, 'time_column', 'posted_date_time')
                chunk[time_column] = pd.to_datetime(
                    chunk[time_column], 
                    format='mixed',
                    utc=True,
                    errors='coerce'
                )
                
                # Filter by date range (keep data between start_time and current_time)
                date_mask = (chunk[time_column] >= start_time) & (chunk[time_column] <= current_time)
                filtered_chunk = chunk[date_mask]
                
                if not filtered_chunk.empty:
                    # Store chunk in database
                    await self.complete_data_storage.store_data(filtered_chunk)
                    record_count += len(filtered_chunk)
                    logger.info(f"Processed chunk {chunks_processed} with {len(filtered_chunk)} rows (Total: {record_count})")
                else:
                    logger.info(f"Chunk {chunks_processed} had no data within retention period")
                
                # Yield to other tasks
                await asyncio.sleep(0)
            
            # Log final stats
            logger.info(f"Data loading summary:")
            logger.info(f"- Files streamed: {len(csv_files)}")
            logger.info(f"- Chunks processed: {chunks_processed}")
            logger.info(f"- Total records loaded: {record_count}")
            
            if record_count == 0:
                logger.error("No data fetched from S3 within retention period")
                return False
            
            logger.info(f"Successfully processed and stored {record_count} total rows")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data from S3: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def generate_embeddings(self, force_refresh: bool = False) -> bool:
        """
        Generate embeddings from stratified sample.
        
        Args:
            force_refresh: Whether to force regeneration of embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Mark operation start
            await self.state_manager.mark_operation_start("generate_embeddings")
            logger.info(f"Generating embeddings (force_refresh={force_refresh})")
            
            # Check if embeddings exist and force_refresh is False
            if not force_refresh:
                embeddings_exist = await self.embedding_storage.embeddings_exist()
                
                if embeddings_exist:
                    embeddings, thread_map = await self.embedding_storage.get_embeddings()
                    
                    if embeddings is not None and thread_map is not None:
                        logger.info("Using existing embeddings (force_refresh=False)")
                        
                        # Mark operation complete
                        await self.state_manager.mark_operation_complete(
                            "generate_embeddings", 
                            {"result": "using_existing"}
                        )
                        
                        return True
            
            # Get stratified sample
            stratified_sample = await self.stratified_storage.get_sample()
            
            if stratified_sample is None or stratified_sample.empty:
                logger.error("No stratified sample available for embedding generation")
                await self.state_manager.update_state({
                    "status": "error",
                    "operation": "generate_embeddings",
                    "error": "Stratified sample is missing or empty"
                })
                return False
            
            # Convert DataFrame to KnowledgeDocument objects for embedding generation
            from knowledge_agents import KnowledgeDocument
            articles = []
            for _, row in stratified_sample.iterrows():
                articles.append(KnowledgeDocument(
                    thread_id=str(row['thread_id']),
                    posted_date_time=str(row.get('posted_date_time', '')),
                    text_clean=str(row.get('text_clean', row.get('content', '')))
                ))
            
            # Determine batch size
            batch_size = getattr(self.config, 'embedding_batch_size', 10)
            
            # Process articles in batches to generate embeddings
            logger.info(f"Processing {len(articles)} articles for embeddings in batches of {batch_size}")
            results = await process_batch(articles, embedding_batch_size=batch_size)
            
            if results:
                # Extract thread IDs and embeddings
                thread_ids = []
                embeddings_list = []
                for result in results:
                    thread_id, _, _, embedding = result
                    if embedding and isinstance(embedding, (list, np.ndarray)):
                        thread_ids.append(str(thread_id))  # Ensure thread_id is a string
                        embeddings_list.append(embedding)
                
                if thread_ids and embeddings_list:
                    # Convert embeddings list to a numpy array
                    embeddings_array = np.array(embeddings_list, dtype=np.float32)
                    
                    # Create thread_id mapping - ensuring thread_id is the key and index is the value
                    thread_id_map = {str(tid): idx for idx, tid in enumerate(thread_ids)}
                    
                    # Log some debug info about the thread map
                    logger.info(f"Created thread_id_map with {len(thread_id_map)} entries")
                    if thread_id_map:
                        sample_keys = list(thread_id_map.keys())[:5]
                        sample_values = [thread_id_map[k] for k in sample_keys]
                        logger.info(f"Sample thread map entries: {list(zip(sample_keys, sample_values))}")
                    
                    # Store embeddings and thread_id mapping
                    await self.embedding_storage.store_embeddings(embeddings_array, thread_id_map)
                    
                    logger.info(f"Successfully generated and stored embeddings with shape {embeddings_array.shape}")
                    
                    # Mark operation complete
                    await self.state_manager.mark_operation_complete(
                        "generate_embeddings", 
                        {
                            "result": "success",
                            "embedding_shape": embeddings_array.shape
                        }
                    )
                    
                    return True
                else:
                    logger.warning("No valid embeddings generated")
                    await self.state_manager.update_state({
                        "status": "warning",
                        "operation": "generate_embeddings",
                        "warning": "No valid embeddings generated"
                    })
                    return False
            else:
                logger.warning("No results returned from batch processing")
                await self.state_manager.update_state({
                    "status": "warning",
                    "operation": "generate_embeddings",
                    "warning": "No results from batch processing"
                })
                return False
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            
            # Mark operation as failed
            await self.state_manager.update_state({
                "status": "error",
                "operation": "generate_embeddings",
                "error": str(e)
            })
            
            return False
    
    async def query_relevant_threads(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query for relevant threads based on embeddings.
        
        Args:
            query_text: Query text to search for
            top_k: Number of top results to return
            
        Returns:
            List of relevant thread IDs and content
        """
        logger.info(f"Querying for relevant threads: {query_text}")
        
        try:
            # Get embeddings and thread ID map
            embeddings, thread_map = await self.embedding_storage.get_embeddings()
            
            if embeddings is None or thread_map is None:
                logger.error("No embeddings available for query")
                raise ValueError("Embeddings not found - please generate embeddings first")
            
            # Get stratified sample for content
            stratified_sample = await self.stratified_storage.get_sample()
            
            if stratified_sample is None:
                logger.error("No stratified sample available for query")
                raise ValueError("Stratified sample not found - please generate stratified sample first")
            
            # Ensure necessary fields are present in the DataFrame
            if "text_clean" not in stratified_sample.columns:
                if "content" in stratified_sample.columns:
                    stratified_sample["text_clean"] = stratified_sample["content"]
                else:
                    logger.error("Stratified sample missing required text field")
                    raise ValueError("Stratified sample missing text_clean or content field")
            
            # Add embeddings to the DataFrame so we can use strings_ranked_by_relatedness
            stratified_sample["embedding"] = None
            
            # Get all thread IDs from the stratified sample
            sample_thread_ids = stratified_sample["thread_id"].astype(str).tolist()
            
            # Count how many thread IDs from the sample are in the thread_map
            matching_ids = [tid for tid in sample_thread_ids if tid in thread_map]
            logger.info(f"Found {len(matching_ids)} out of {len(sample_thread_ids)} thread IDs in the embedding thread map")
            
            # Efficiently assign embeddings using vectorized operations where possible
            for idx, row in stratified_sample.iterrows():
                thread_id = str(row["thread_id"])
                if thread_id in thread_map:
                    emb_idx = thread_map[thread_id]
                    if isinstance(emb_idx, (int, str)) and str(emb_idx).isdigit():
                        emb_idx = int(emb_idx)
                        if 0 <= emb_idx < len(embeddings):
                            stratified_sample.at[idx, "embedding"] = embeddings[emb_idx]
            
            # Check if we need to try reverse mapping
            valid_embeddings_count = stratified_sample["embedding"].count()
            if valid_embeddings_count < len(stratified_sample) * 0.5:  # If less than 50% matched
                logger.warning("Less than 50% of thread IDs matched directly, trying reverse mapping")
                thread_map_reversed = {v: k for k, v in thread_map.items()}
                
                # Try reverse mapping for rows without embeddings
                for idx, row in stratified_sample[stratified_sample["embedding"].isna()].iterrows():
                    thread_id = str(row["thread_id"])
                    if thread_id in thread_map_reversed:
                        emb_idx = thread_map_reversed[thread_id]
                        if isinstance(emb_idx, (int, str)) and str(emb_idx).isdigit():
                            emb_idx = int(emb_idx)
                            if 0 <= emb_idx < len(embeddings):
                                stratified_sample.at[idx, "embedding"] = embeddings[emb_idx]
            
            # Final verification
            valid_embeddings_count = stratified_sample["embedding"].count()
            if valid_embeddings_count == 0:
                logger.error("No valid embeddings could be matched to thread IDs")
                logger.error(f"Thread map contains {len(thread_map)} entries, sample contains {len(stratified_sample)} rows")
                # Add debug info for first few thread IDs
                sample_thread_ids = stratified_sample["thread_id"].head(5).tolist()
                logger.error(f"Sample thread IDs (first 5): {sample_thread_ids}")
                thread_map_keys = list(thread_map.keys())[:5]
                logger.error(f"Thread map keys (first 5): {thread_map_keys}")
                raise ValueError("No embeddings could be matched to thread IDs - check thread ID consistency")
            else:
                logger.info(f"Successfully matched {valid_embeddings_count} embeddings out of {len(stratified_sample)} stratified sample rows")
            
            # Import inference functions only when needed to avoid circular imports
            from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider
            
            # Create a proper embedding function that uses our pre-computed embeddings
            def embedding_function(text):
                # This is just a placeholder that returns a numpy array of the correct size
                # The actual embeddings come from our pre-computed data
                return np.zeros(embeddings.shape[1])
            
            # Initialize a model provider with the embedding function
            # ModelProvider is an Enum and doesn't accept keyword arguments
            # provider = ModelProvider(
            #     embedding_provider=embedding_function
            # )
            
            # Fix: Use None instead since ModelProvider is an Enum
            provider = None
            
            # Create a simple agent that uses our provider
            dummy_agent = KnowledgeAgent()
            
            # Lazy import strings_ranked_by_relatedness to avoid circular dependencies
            from knowledge_agents.inference_ops import strings_ranked_by_relatedness
            
            # Get relevant threads using proper vector search
            ranked_results = await strings_ranked_by_relatedness(
                query=query_text,
                df=stratified_sample,
                agent=dummy_agent,
                top_n=top_k,
                provider=provider
            )
            
            # Format results
            result_threads = []
            for text, similarity, metadata in ranked_results:
                result_threads.append({
                    "thread_id": metadata.get("thread_id"),
                    "content": text,
                    "posted_date_time": metadata.get("posted_date_time"),
                    "similarity": float(similarity)
                })
            
            return result_threads
            
        except Exception as e:
            logger.error(f"Error querying relevant threads: {e}")
            raise
    
    async def is_data_ready(self, skip_embeddings: bool = False) -> bool:
        """
        Check if data is ready for use.
        
        Args:
            skip_embeddings: Whether to skip checking embeddings
            
        Returns:
            True if data is ready, False otherwise
        """
        try:
            # Check if complete data exists
            row_count = await self.complete_data_storage.get_row_count()
            if row_count == 0:
                logger.warning("Complete data is empty")
                return False
            
            # Check if stratified sample exists
            stratified_exists = await self.stratified_storage.sample_exists()
            if not stratified_exists:
                logger.warning("Stratified sample does not exist")
                return False
            
            # Check if embeddings exist (if not skipped)
            if not skip_embeddings:
                embeddings_exist = await self.embedding_storage.embeddings_exist()
                if not embeddings_exist:
                    logger.warning("Embeddings do not exist")
                    return False
            
            # Check if any operation is in progress
            operation_in_progress = await self.state_manager.is_operation_in_progress("ensure_data_ready") or \
                                   await self.state_manager.is_operation_in_progress("generate_embeddings")
            
            if operation_in_progress:
                logger.info("Data operations are in progress")
                return False
            
            logger.info("Data is ready for use")
            return True
            
        except Exception as e:
            logger.error(f"Error checking if data is ready: {e}")
            return False
    
    async def create_stratified_sample(self, force_refresh: bool = False) -> bool:
        """
        Create stratified sample for data.
        
        Args:
            force_refresh: Whether to force the recreation of the stratified sample
            
        Returns:
            bool: True if stratified sample is ready, False otherwise
        """
        logger.info(f"Creating stratified sample (force_refresh={force_refresh})")
        
        try:
            # Check if we have a stratified sample and don't need to refresh
            if not force_refresh:
                stratified_sample = await self.stratified_storage.get_sample()
                
                # If we have a sample, check if it's too old
                if stratified_sample is not None and not stratified_sample.empty:
                    # Check the date range of the sample
                    if self.config.time_column in stratified_sample.columns:
                        max_date = stratified_sample[self.config.time_column].max()
                        if isinstance(max_date, pd.Timestamp):
                            current_time = datetime.now(timezone.utc)
                            sample_age_hours = (current_time - max_date.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 3600
                            
                            # If sample is more than 24 hours old, force a refresh
                            if sample_age_hours > 24:
                                logger.info(f"Stratified sample is {sample_age_hours:.2f} hours old, forcing refresh")
                                force_refresh = True
                            else:
                                logger.info(f"Using existing stratified sample with {len(stratified_sample)} rows (age: {sample_age_hours:.2f} hours)")
                                return True
                    else:
                        logger.info(f"Using existing stratified sample with {len(stratified_sample)} rows")
                        return True
            
            # Get complete data
            logger.info("No stratified sample found or force_refresh enabled, loading complete data")
            logger.info(f"Using filter_date: {self.config.filter_date}")
            complete_data = await self.complete_data_storage.get_data(self.config.filter_date)
            
            if complete_data is None or complete_data.empty:
                logger.warning("Complete data is empty, cannot create stratified sample")
                return False
                
            # Add diagnostic logging for date range
            if self.config.time_column in complete_data.columns:
                min_date = complete_data[self.config.time_column].min()
                max_date = complete_data[self.config.time_column].max()
                logger.info(f"Complete data date range: {min_date} to {max_date}")
                logger.info(f"Current date: {datetime.now()} (UTC: {datetime.now(timezone.utc)})")
                logger.info(f"Data age: most recent data is {(datetime.now(timezone.utc) - max_date.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 3600:.2f} hours old")
            
            logger.info(f"Creating stratified sample from {len(complete_data)} rows")
            
            # Create stratified sample
            try:
                # Set sample size
                sample_size = min(self.config.sample_size, len(complete_data))
                
                logger.info(f"Using sampler.py implementation for time-based stratified sampling")
                # Use the sampler module's implementation which properly handles datetime conversion
                stratified = self.sampler.stratified_sample(complete_data)
                
                # If the result is too small or empty, fall back to random sampling
                if stratified is None or stratified.empty or len(stratified) < min(sample_size, len(complete_data) * 0.1):
                    logger.warning(f"Sampler returned too few samples ({0 if stratified is None or stratified.empty else len(stratified)}), falling back to random sampling")
                    stratified = complete_data.sample(min(sample_size, len(complete_data)))
                
                # Ensure all columns are serializable
                safe_stratified = stratified.copy()
                for col in safe_stratified.columns:
                    if pd.api.types.is_datetime64_dtype(safe_stratified[col]):
                        safe_stratified[col] = safe_stratified[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                    elif pd.api.types.is_object_dtype(safe_stratified[col]):
                        # Ensure all object columns are properly serializable strings
                        safe_stratified[col] = safe_stratified[col].astype(str)
                
                # Reset index for clean serialization
                safe_stratified = safe_stratified.reset_index(drop=True)
                
                # Store stratified sample
                storage_success = await self.stratified_storage.store_sample(safe_stratified)
                
                if storage_success:
                    logger.info(f"Successfully created and stored stratified sample with {len(safe_stratified)} rows")
                    return True
                else:
                    logger.error(f"Failed to store stratified sample")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating stratified sample: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in create_stratified_sample: {e}")
            return False 