"""Knowledge agents runner module."""
import logging
import asyncio
from typing import Tuple, List, Union
from . import KnowledgeAgentConfig
from .model_ops import ModelProvider, ModelOperation, KnowledgeAgent
from .data_ops import DataConfig, DataOperations
from .inference_ops import summarize_text
from .embedding_ops import get_relevant_content
from config.settings import Config
import nest_asyncio
import IPython
import os

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
    force_refresh: bool = False,
) -> Tuple[List[str], str]:
    """Async implementation of knowledge agents pipeline with three distinct models."""
    try:
        logger.info("Starting knowledge agents pipeline")
        
        # Create knowledge agent
        agent = KnowledgeAgent()
        
        # Initialize data operations with proper configuration
        data_config = DataConfig(
            root_path=config.root_path,
            all_data_path=config.all_data_path,
            stratified_data_path=config.stratified_data_path,
            knowledge_base_path=config.knowledge_base_path,
            sample_size=config.sample_size,
            filter_date=os.getenv('FILTER_DATE'),
            time_column=os.getenv('TIME_COLUMN', 'posted_date_time'),
            strata_column=os.getenv('STRATA_COLUMN', None)
        )
        data_ops = DataOperations(data_config)
        
        # Prepare data using new data operations
        try:
            logger.info("Preparing data...")
            await data_ops.prepare_data(force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
        
        # Step 1: Generate embeddings using embedding model
        logger.info(f"Using {config.providers[ModelOperation.EMBEDDING]} for embeddings")
        try:
            get_relevant_content(
                library=str(config.stratified_data_path),
                knowledge_base=str(config.knowledge_base_path),
                batch_size=config.batch_size,
                provider=config.providers[ModelOperation.EMBEDDING]
            )
        except Exception as e:
            logger.error(f"Error getting relevant content: {e}")
            raise
        
        # Step 2 & 3: Generate chunks/context and final summary
        logger.info(f"Using {config.providers[ModelOperation.CHUNK_GENERATION]} for chunk analysis")
        logger.info(f"Using {config.providers[ModelOperation.SUMMARIZATION]} for final summary")
        try:
            chunks, response = await summarize_text(
                query=query,
                agent=agent,
                knowledge_base_path=str(config.knowledge_base_path),
                batch_size=config.batch_size,
                max_workers=config.max_workers,
                providers=config.providers
            )
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
) -> Union[Tuple[List[str], str], "asyncio.Future"]:
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
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Run knowledge agents with three-model pipeline')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of data and knowledge base')
    
    # Use Config class defaults for optional arguments
    parser.add_argument('--batch-size', type=int, default=Config.DEFAULT_BATCH_SIZE, 
                       help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, default=Config.DEFAULT_MAX_WORKERS,
                       help='Maximum number of worker threads')
    parser.add_argument('--root-path', type=str, default=Config.ROOT_PATH,
                       help='Root path for data storage')
    parser.add_argument('--sample-size', type=int, default=Config.DEFAULT_BATCH_SIZE,
                       help='Sample size for data processing')
    
    # Add provider arguments with Config defaults
    provider_settings = Config.get_provider_settings()
    parser.add_argument('--embedding-provider', type=str, 
                       choices=['openai', 'grok', 'venice'],
                       default=provider_settings['embedding_provider'],
                       help='Provider for embeddings')
    parser.add_argument('--chunk-provider', type=str,
                       choices=['openai', 'grok', 'venice'],
                       default=provider_settings['chunk_provider'],
                       help='Provider for chunk generation and context analysis')
    parser.add_argument('--summary-provider', type=str,
                       choices=['openai', 'grok', 'venice'],
                       default=provider_settings['summary_provider'],
                       help='Provider for final analysis and forecasting')
    
    args = parser.parse_args()
    
    try:
        # Build providers dictionary from arguments
        providers = {
            ModelOperation.EMBEDDING: ModelProvider(args.embedding_provider),
            ModelOperation.CHUNK_GENERATION: ModelProvider(args.chunk_provider),
            ModelOperation.SUMMARIZATION: ModelProvider(args.summary_provider)
        }
            
        # Get paths from Config
        paths = Config.get_data_paths()
        root_path = Path(args.root_path)
        
        # Create configuration using Config class values
        config = KnowledgeAgentConfig(
            root_path=root_path,
            all_data_path=paths['all_data'],
            stratified_data_path=paths['stratified'],
            knowledge_base_path=paths['knowledge_base'],
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            providers=providers
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
        
    except Exception as e:
        logger.error(f"Error running knowledge agents: {e}")
        raise

if __name__ == "__main__":
    main()
