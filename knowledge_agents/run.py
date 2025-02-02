"""Knowledge agents runner module."""
import logging
import asyncio
from typing import Tuple, List, Union, Dict, Any
from . import KnowledgeAgentConfig
from .model_ops import ModelProvider, ModelOperation, KnowledgeAgent
from .data_ops import DataConfig, prepare_knowledge_base
from .inference_ops import process_multiple_queries
from .embedding_ops import get_relevant_content
from config.settings import Config
import nest_asyncio
import IPython
import os
from pathlib import Path

# Enable nested asyncio for Jupyter notebooks
try:
    nest_asyncio.apply()
except Exception:
    pass

# Initialize logging with IPython-friendly format
class IPythonFormatter(logging.Formatter):
    """Custom formatter that detects if we're in a notebook."""
    def format(self, record):
        if IPython.get_ipython() is not None:
            # In notebook - use simple format
            self._style._fmt = "%(message)s"
        else:
            # In terminal - use detailed format
            self._style._fmt = "%(asctime)s - %(levelname)s - %(message)s"
        return super().format(record)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

async def _run_knowledge_agents_async(
    query: str,
    config: KnowledgeAgentConfig,
    force_refresh: bool = False
) -> Tuple[List[Dict[str, Any]], str]:
    """Async implementation of knowledge agents pipeline with three distinct models."""
    try:
        logger.info("Starting knowledge agents pipeline")

        # Create knowledge agent
        agent = KnowledgeAgent()

        # Initialize data operations with proper configuration
        data_config = DataConfig(
            root_data_path=config.root_data_path,
            stratified_data_path=config.stratified_data_path,
            knowledge_base_path=config.knowledge_base_path,
            temp_path=Path(Config.get_paths()['temp']),
            filter_date=os.getenv('FILTER_DATE')
        )

        # Prepare data and process references using the full pipeline
        try:
            logger.info("Preparing knowledge base...")
            result = await prepare_knowledge_base(force_refresh=force_refresh)
            logger.info(f"Knowledge base preparation result: {result}")
        except Exception as e:
            logger.error(f"Error preparing knowledge base: {e}")
            raise

        # Step 1: Generate embeddings using embedding model
        logger.info(f"Using {config.providers[ModelOperation.EMBEDDING]} for embeddings")
        try:
            await get_relevant_content(
                library=str(config.stratified_data_path),
                knowledge_base=str(config.knowledge_base_path),
                batch_size=config.embedding_batch_size,
                provider=config.providers[ModelOperation.EMBEDDING],
                force_refresh=force_refresh
            )
        except Exception as e:
            logger.error(f"Error getting relevant content: {e}")
            raise

        # Step 2 & 3: Generate chunks/context and final summary
        logger.info(f"Using {config.providers[ModelOperation.CHUNK_GENERATION]} for chunk analysis")
        logger.info(f"Using {config.providers[ModelOperation.SUMMARIZATION]} for final summary")
        try:
            # Process single query using batch processing function
            results = await process_multiple_queries(
                queries=[query],  # Wrap single query in list
                agent=agent,
                knowledge_base_path=str(config.knowledge_base_path),
                chunk_batch_size=config.chunk_batch_size,
                summary_batch_size=config.summary_batch_size,
                max_workers=config.max_workers,
                providers=config.providers
            )
            # Unpack results for single query case
            chunks, response = results[0]
            logger.info("Summary generated successfully")
            return chunks, response

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    except Exception as e:
        logger.error(f"Error in knowledge agent pipeline: {e}")
        raise

def run_knowledge_agents(
    query: str,
    config: KnowledgeAgentConfig,
    force_refresh: bool = False,
) -> Union[Tuple[List[Dict[str, Any]], str], "asyncio.Future"]:
    """Run knowledge agents pipeline in both notebook and script environments.

    This function detects the environment and handles the async execution appropriately.
    In a notebook, it will execute the coroutine immediately.
    In a script, it will return the coroutine for the event loop to execute.

    Args:
        query: The search query
        config: Configuration instance for the knowledge agents
        force_refresh: Whether to force refresh the data

    Returns:
        In notebook: Tuple of (processed chunks, final summary)
        In script: Coroutine object
    """
    coroutine = _run_knowledge_agents_async(
        query=query,
        config=config,
        force_refresh=force_refresh,
    )

    # If we're in a notebook, execute the coroutine immediately
    if IPython.get_ipython() is not None:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)

    # Otherwise, return the coroutine for the script's event loop
    return coroutine

def main():
    """Main entry point with support for three-model pipeline selection."""
    import argparse
    parser = argparse.ArgumentParser(description='Run knowledge agents pipeline')
    parser.add_argument('--query', type=str, required=True,
                      help='Query to process')
    parser.add_argument('--force-refresh', action='store_true',
                      help='Force refresh of cached results')
    
    # Get settings from Config
    sample_settings = Config.get_sample_settings()
    model_settings = Config.get_model_settings()
    processing_settings = Config.get_processing_settings()
    paths = Config.get_paths()
    
    parser.add_argument('--sample-size', type=int, default=sample_settings['default_sample_size'],
                      help='Number of samples to process')
    parser.add_argument('--embedding-batch-size', type=int, default=model_settings['embedding_batch_size'],
                      help='Batch size for embedding operations')
    parser.add_argument('--chunk-batch-size', type=int, default=model_settings['chunk_batch_size'],
                      help='Batch size for chunk generation')
    parser.add_argument('--summary-batch-size', type=int, default=model_settings['summary_batch_size'],
                      help='Batch size for summary generation')
    parser.add_argument('--max-workers', type=int, default=processing_settings['max_workers'],
                      help='Maximum number of workers for parallel processing')
    args = parser.parse_args()

    # Create configuration
    config = KnowledgeAgentConfig(
        root_data_path=paths['root_data_path'],
        stratified_data_path=paths['stratified'],
        knowledge_base_path=paths['knowledge_base'],
        sample_size=args.sample_size,
        embedding_batch_size=args.embedding_batch_size,
        chunk_batch_size=args.chunk_batch_size,
        summary_batch_size=args.summary_batch_size,
        max_workers=args.max_workers,
        providers={
            ModelOperation.EMBEDDING: ModelProvider(model_settings['default_embedding_provider']),
            ModelOperation.CHUNK_GENERATION: ModelProvider(model_settings['default_chunk_provider']),
            ModelOperation.SUMMARIZATION: ModelProvider(model_settings['default_summary_provider'])
        }
    )

    # Run the pipeline
    loop = asyncio.get_event_loop()
    chunks, response = loop.run_until_complete(
        _run_knowledge_agents_async(
            query=args.query,
            config=config,
            force_refresh=args.force_refresh
        )
    )

    print("\nRelevant Chunks:")
    print("-" * 80)
    for chunk in chunks:
        print(chunk)
        print("-" * 80)

    print("\nGenerated Summary:")
    print("-" * 80)
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main()