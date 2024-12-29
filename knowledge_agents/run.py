import logging
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from .model_ops import ModelProvider, ModelOperation
from .data_ops import prepare_data
from .inference_ops import summarize_text
from .embedding_ops import get_relevant_content
import nest_asyncio
import IPython

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
handler.setFormatter(IPythonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_run_config(config_path: str = './config.ini') -> Dict[str, str]:
    """Load and validate configuration settings from environment variables."""
    try:
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        required_settings = {
            'ROOT_PATH': os.getenv('ROOT_PATH', './data'),
            'ALL_DATA': os.getenv('ALL_DATA', './data/all_data.csv'),
            'ALL_DATA_STRATIFIED_PATH': os.getenv('ALL_DATA_STRATIFIED_PATH', './data/stratified'),
            'KNOWLEDGE_BASE': os.getenv('KNOWLEDGE_BASE', './data/knowledge_base.csv'),
            'SAMPLE_SIZE': os.getenv('SAMPLE_SIZE', '2500'),
            'FILTER_DATE': os.getenv('FILTER_DATE', '2024-12-26')
        }
        
        # Create directories if they don't exist
        for key in ['ROOT_PATH', 'ALL_DATA_STRATIFIED_PATH']:
            path = Path(required_settings[key])
            path.mkdir(parents=True, exist_ok=True)
        
        # Ensure parent directory of KNOWLEDGE_BASE exists
        knowledge_base_path = Path(required_settings['KNOWLEDGE_BASE'])
        knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        
        return required_settings
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def validate_providers(providers: Optional[Dict[ModelOperation, ModelProvider]] = None) -> Dict[ModelOperation, ModelProvider]:
    """Validate and set default providers for the three-model pipeline."""
    if providers is None:
        providers = {}
    
    # Ensure we have all required operations
    if ModelOperation.EMBEDDING not in providers:
        providers[ModelOperation.EMBEDDING] = ModelProvider.OPENAI
        logger.info(f"Using default embedding provider: {providers[ModelOperation.EMBEDDING]}")
    
    if ModelOperation.CHUNK_GENERATION not in providers:
        providers[ModelOperation.CHUNK_GENERATION] = ModelProvider.GROK
        logger.info(f"Using default chunking provider: {providers[ModelOperation.CHUNK_GENERATION]}")
    
    if ModelOperation.SUMMARIZATION not in providers:
        providers[ModelOperation.SUMMARIZATION] = ModelProvider.VENICE
        logger.info(f"Using default summarization provider: {providers[ModelOperation.SUMMARIZATION]}")
    
    # Validate embedding provider (only OpenAI and Grok support embeddings)
    if providers[ModelOperation.EMBEDDING] not in [ModelProvider.OPENAI, ModelProvider.GROK]:
        logger.warning(f"Unsupported embedding provider: {providers[ModelOperation.EMBEDDING]}. Falling back to OpenAI.")
        providers[ModelOperation.EMBEDDING] = ModelProvider.OPENAI
    
    return providers

async def _run_knowledge_agents_async(
    query: str,
    process_new: bool = False,
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None
) -> Tuple[List[str], str]:
    """Async implementation of knowledge agents pipeline with three distinct models."""
    try:
        # Validate and set default providers
        providers = validate_providers(providers)
        
        # Load configuration
        config = load_run_config()
        logger.info("Configuration loaded successfully")
        
        # Prepare data if requested
        if process_new:
            logger.info("Processing new data...")
            try:
                prepare_data(process_new=True)
            except Exception as e:
                logger.error(f"Error preparing data: {e}")
                raise
        
        # Step 1: Generate embeddings using embedding model
        logger.info(f"Using {providers[ModelOperation.EMBEDDING]} for embeddings")
        try:
            get_relevant_content(
                library=config['ALL_DATA_STRATIFIED_PATH'],
                knowledge_base=config['KNOWLEDGE_BASE'],
                batch_size=batch_size,
                provider=providers[ModelOperation.EMBEDDING]
            )
        except Exception as e:
            logger.error(f"Error getting relevant content: {e}")
            raise
        
        # Step 2 & 3: Generate chunks/context and final summary
        logger.info(f"Using {providers[ModelOperation.CHUNK_GENERATION]} for chunk analysis")
        logger.info(f"Using {providers[ModelOperation.SUMMARIZATION]} for final summary")
        try:
            chunks, response = await summarize_text(
                query=query,
                knowledge_base_path=config['KNOWLEDGE_BASE'],
                batch_size=batch_size,
                max_workers=max_workers,
                providers=providers
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
    process_new: bool = False,
    batch_size: int = 100,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None
) -> Union[Tuple[List[str], str], asyncio.coroutine]:
    """Run knowledge agents pipeline in both notebook and script environments.
    
    This function detects the environment and handles the async execution appropriately.
    In a notebook, it will execute the coroutine immediately.
    In a script, it will return the coroutine for the event loop to execute.
    
    Args:
        query: The search query
        process_new: Whether to process new data
        config_path: Path to configuration file
        batch_size: Size of batches for processing
        max_workers: Maximum number of worker threads
        providers: Dictionary mapping operations to providers
            Example: {
                ModelOperation.EMBEDDING: ModelProvider.OPENAI,
                ModelOperation.CHUNK_GENERATION: ModelProvider.GROK,
                ModelOperation.SUMMARIZATION: ModelProvider.VENICE
            }
    
    Returns:
        In notebook: Tuple of (processed chunks, final summary)
        In script: Coroutine object
    """
    coroutine = _run_knowledge_agents_async(
        query=query,
        process_new=process_new,
        batch_size=batch_size,
        max_workers=max_workers,
        providers=providers
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
    parser = argparse.ArgumentParser(description='Run knowledge agents with three-model pipeline')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--process-new', action='store_true', help='Process new data')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of worker threads')
    
    # Add provider arguments for each operation
    parser.add_argument('--embedding-provider', type=str, choices=['openai', 'grok'],
                      help='Provider for embeddings (only OpenAI and Grok supported)')
    parser.add_argument('--chunk-provider', type=str, choices=['openai', 'grok', 'venice'],
                      help='Provider for chunk generation and context analysis')
    parser.add_argument('--summary-provider', type=str, choices=['openai', 'grok', 'venice'],
                      help='Provider for final analysis and forecasting')
    
    args = parser.parse_args()
    
    try:
        # Build providers dictionary from arguments
        providers = {}
        if args.embedding_provider:
            providers[ModelOperation.EMBEDDING] = ModelProvider(args.embedding_provider)
        if args.chunk_provider:
            providers[ModelOperation.CHUNK_GENERATION] = ModelProvider(args.chunk_provider)
        if args.summary_provider:
            providers[ModelOperation.SUMMARIZATION] = ModelProvider(args.summary_provider)
        
        # Validate providers
        providers = validate_providers(providers)
        
        # Get the coroutine and run it in the event loop
        coroutine = run_knowledge_agents(
            query=args.query,
            process_new=args.process_new,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            providers=providers
        )
        chunks, response = asyncio.run(coroutine)
        
        print("\nGenerated Summary:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"Error running knowledge agents: {e}")
        raise

if __name__ == "__main__":
    main()