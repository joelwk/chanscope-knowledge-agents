import logging
import asyncio
from typing import Tuple, List, Union
from . import KnowledgeAgentConfig, ModelOperation, ModelProvider
from .model_ops import ModelConfig, KnowledgeAgent
from .data_ops import prepare_data
from .inference_ops import summarize_text
from .embedding_ops import get_relevant_content
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
    process_new: bool = False,
) -> Tuple[List[str], str]:
    """Async implementation of knowledge agents pipeline with three distinct models."""
    try:
        logger.info("Starting knowledge agents pipeline")
        
        # Create model config from environment variables
        model_config = ModelConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_completion_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            grok_api_key=os.getenv("GROK_API_KEY"),
            grok_embedding_model=os.getenv("GROK_EMBEDDING_MODEL", "grok-v1-embedding"),
            grok_completion_model=os.getenv("GROK_MODEL", "grok-2-1212"),
            venice_api_key=os.getenv("VENICE_API_KEY"),
            venice_summary_model=os.getenv("VENICE_MODEL", "llama-3.1-405b"),
            venice_chunk_model=os.getenv("VENICE_CHUNK_MODEL", "dolphin-2.9.2-qwen2-72b"),
            default_embedding_provider=config.providers.get(ModelOperation.EMBEDDING, ModelProvider.OPENAI)
        )
        
        # Create knowledge agent
        agent = KnowledgeAgent(model_config)
        
        # Prepare data if requested
        if process_new:
            logger.info("Processing new data...")
            try:
                # Set environment variables for data_ops
                os.environ['ROOT_PATH'] = str(config.root_path)
                os.environ['ALL_DATA'] = str(config.all_data_path)
                os.environ['ALL_DATA_STRATIFIED_PATH'] = str(config.stratified_data_path)
                os.environ['SAMPLE_SIZE'] = str(config.sample_size)
                
                prepare_data(process_new=True)
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
    process_new: bool = False,
) -> Union[Tuple[List[str], str], "asyncio.Future"]:
    """Run knowledge agents pipeline in both notebook and script environments.
    
    This function detects the environment and handles the async execution appropriately.
    In a notebook, it will execute the coroutine immediately.
    In a script, it will return the coroutine for the event loop to execute.
    
    Args:
        query: The search query
        config: Configuration instance for the knowledge agents
        process_new: Whether to process new data
    
    Returns:
        In notebook: Tuple of (processed chunks, final summary)
        In script: Coroutine object
    """
    coroutine = _run_knowledge_agents_async(
        query=query,
        config=config,
        process_new=process_new,
    )
    
    # If we're in a notebook, execute the coroutine immediately
    if IPython.get_ipython() is not None:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)
    
    # Otherwise, return the coroutine for the script's event loop
    return coroutine
