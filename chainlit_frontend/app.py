import chainlit as cl
import logging
import os
from chainlit.input_widget import Select, Switch, Slider, TextInput
import aiohttp
from aiohttp import ClientTimeout
import asyncio
from chainlit_frontend import API_BASE_URL
from datetime import datetime
from config.settings import Config
from knowledge_agents.model_ops import ModelProvider

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Get API settings
api_settings = Config.get_api_settings()
API_PORT = api_settings.get('port')
API_BASE_URL = api_settings.get('base_url')
API_TIMEOUT = ClientTimeout(total=1200, connect=120)  # Increase total timeout to 10 minutes

# Valid providers from ModelProvider enum
VALID_PROVIDERS = [provider.value for provider in ModelProvider]

# Environment-specific configurations
IS_REPLIT = bool(os.getenv('REPL_SLUG') and os.getenv('REPL_OWNER'))
IS_DOCKER = Config.is_docker_env()

def get_base_urls():
    """Get appropriate base URLs based on environment."""
    if IS_REPLIT:
        return ['http://0.0.0.0:5000']
    elif IS_DOCKER:
        return [
            'http://api:5000',
            'http://0.0.0.0:5000',
            'http://localhost:5000'
        ]
    return [
        'http://0.0.0.0:5000',
        'http://localhost:5000'
    ]

def prepare_settings_for_api(settings: dict) -> dict:
    """Prepare settings for API consumption based on environment."""
    # Get paths from Config
    path_settings = Config.PATH_SETTINGS
    
    # Base settings that are always included
    api_settings = {
        "force_refresh": settings.get("force_refresh", False),
        "filter_date": settings.get("filter_date"),
        "sample_size": settings.get("sample_size"),
        "embedding_provider": settings.get("embedding_provider"),
        "chunk_provider": settings.get("chunk_provider"),
        "summary_provider": settings.get("summary_provider"),
        
        # Processing settings
        "max_workers": settings.get("max_workers"),
        "cache_enabled": settings.get("cache_enabled"),
        "padding_enabled": settings.get("padding_enabled", False),
        "contraction_mapping_enabled": settings.get("contraction_mapping_enabled", False),
        "non_alpha_numeric_enabled": settings.get("non_alpha_numeric_enabled", False),
        "max_tokens": settings.get("max_tokens"),
        
        # Batch settings
        "embedding_batch_size": settings.get("embedding_batch_size"),
        "chunk_batch_size": settings.get("chunk_batch_size"),
        "summary_batch_size": settings.get("summary_batch_size"),
        "chunk_size": settings.get("chunk_size"),
        "processing_chunk_size": settings.get("processing_chunk_size"),
        "stratification_chunk_size": settings.get("stratification_chunk_size")
    }
    
    # Add path settings based on environment
    if IS_DOCKER:
        app_prefix = "/app"
        api_settings.update({
            "root_data_path": os.path.join(app_prefix, "data"),
            "stratified_path": os.path.join(app_prefix, "data", "stratified"),
            "knowledge_base_path": os.path.join(app_prefix, "data", "knowledge_base.csv"),
            "temp_path": os.path.join(app_prefix, "temp_files")
        })
    elif IS_REPLIT:
        api_settings.update({
            "root_data_path": path_settings['root_data_path'],
            "stratified_path": os.path.join(path_settings['root_data_path'], 'stratified'),
            "knowledge_base_path": os.path.join(path_settings['root_data_path'], 'knowledge_base.csv'),
            "temp_path": "temp_files"
        })
    else:
        # Local development
        api_settings.update({
            "root_data_path": path_settings['root_data_path'],
            "stratified_path": path_settings['stratified'],
            "knowledge_base_path": path_settings['knowledge_base'],
            "temp_path": path_settings['temp']
        })
    
    return api_settings

async def process_query(query: str, settings: dict):
    """Process a query using the API endpoint."""
    # Get environment-specific base URLs
    base_urls = get_base_urls()
    
    logger.info("=== API Connection Attempt ===")
    logger.info(f"Environment: {'Replit' if IS_REPLIT else 'Docker' if IS_DOCKER else 'Local'}")
    logger.info(f"Available API endpoints: {base_urls}")
    
    # Prepare settings for API
    api_settings = prepare_settings_for_api(settings)
    logger.info(f"Using settings: {api_settings}")

    # Create session with proper connection settings
    connector = aiohttp.TCPConnector(
        force_close=False,
        keepalive_timeout=300,
        enable_cleanup_closed=True
    )

    async with aiohttp.ClientSession(
        timeout=API_TIMEOUT,
        connector=connector
    ) as session:
        logger.info("Created aiohttp session")

        # Try each base URL in order
        for base_url in base_urls:
            try:
                # Add /api prefix for Replit environment
                api_prefix = "/api" if IS_REPLIT else ""
                logger.info(f"Attempting to connect to API at {base_url}")
                
                async with session.post(
                    f"{base_url}{api_prefix}/process_query",
                    json={
                        "query": query,
                        **api_settings
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                ) as response:
                    logger.info(f"API Response status: {response.status}")
                    response_text = await response.text()
                    logger.info(f"API Response body: {response_text}")

                    if response.status != 200:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("message", "API request failed")
                        except:
                            error_msg = response_text
                        raise Exception(f"API request failed with status {response.status}: {error_msg}")

                    try:
                        data = await response.json()
                        return data["results"]["chunks"], data["results"]["summary"]
                    except Exception as e:
                        logger.error(f"Error parsing API response: {str(e)}")
                        raise Exception(f"Invalid API response format: {response_text}")

            except Exception as e:
                logger.error(f"API connection error for {base_url}: {str(e)}")
                continue

        # If we get here, all URLs failed
        logger.error("All API endpoints failed.")
        raise Exception("Failed to connect to any API endpoint. Please try again in a moment.")

def format_chunk(chunk):
    """Format a chunk for display."""
    if isinstance(chunk, dict):
        # Extract the thread analysis if it exists
        analysis = chunk.get('analysis', {}).get('analysis', {}).get('thread_analysis', 'No analysis available')
        posted_date = chunk.get('posted_date_time', 'No date')
        thread_id = chunk.get('thread_id', 'No ID')
        
        return f"""
Thread ID: {thread_id}
Posted: {posted_date}
Analysis:
{analysis}
-------------------"""
    elif isinstance(chunk, str):
        return chunk
    else:
        return str(chunk)

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    await cl.Message(
        content=
        "Welcome to Chanscope. Before you start, please review the Readme to understand how to use this tool. It is not a chatbot, but a tool for querying 4chan data to generate insights and forecasts."
    ).send()

    # Get all configuration settings
    processing_settings = Config.get_processing_settings()
    model_settings = Config.get_model_settings()
    sample_settings = Config.get_sample_settings()
    chunk_settings = Config.get_chunk_settings()
    path_settings = Config.PATH_SETTINGS

    # Initialize chat settings with all configuration options
    settings = await cl.ChatSettings([
        # Processing settings
        Switch(id="force_refresh",
               label="Force Data Refresh",
               initial=False,
               description="Enable to force refresh the knowledge base data"),
        TextInput(id="filter_date",
                  label="Filter Date (YYYY-MM-DD)",
                  initial=processing_settings['filter_date'],
                  description="Date to filter data from (format: YYYY-MM-DD)"),
        Slider(id="sample_size",
               label="Sample Size",
               initial=sample_settings['default_sample_size'],
               min=sample_settings['min_sample_size'],
               max=sample_settings['max_sample_size'],
               step=10,
               description="Number of items to process in each batch"),
        # Model provider settings
        Select(id="embedding_provider",
               label="Embedding Provider",
               values=VALID_PROVIDERS,
               initial_index=VALID_PROVIDERS.index(model_settings['default_embedding_provider']),
               description="Select the provider for embeddings"),
        Select(id="chunk_provider",
               label="Chunk Provider",
               values=VALID_PROVIDERS,
               initial_index=VALID_PROVIDERS.index(model_settings['default_chunk_provider']),
               description="Select the provider for text chunking"),
        Select(id="summary_provider",
               label="Summary Provider",
               values=VALID_PROVIDERS,
               initial_index=VALID_PROVIDERS.index(model_settings['default_summary_provider']),
               description="Select the provider for summarization")
    ]).send()

    # Store initial settings
    initial_settings = {
        # User configurable settings
        "force_refresh": False,
        "filter_date": processing_settings['filter_date'],
        "sample_size": sample_settings['default_sample_size'],
        
        # Provider settings
        "embedding_provider": model_settings['default_embedding_provider'],
        "chunk_provider": model_settings['default_chunk_provider'],
        "summary_provider": model_settings['default_summary_provider'],
        
        # Processing settings from Config
        "max_workers": processing_settings['max_workers'],
        "cache_enabled": processing_settings['cache_enabled'],
        "padding_enabled": processing_settings['padding_enabled'],
        "contraction_mapping_enabled": processing_settings['contraction_mapping_enabled'],
        "non_alpha_numeric_enabled": processing_settings['non_alpha_numeric_enabled'],
        "max_tokens": processing_settings['max_tokens'],
        
        # Batch and chunk settings from Config
        "embedding_batch_size": model_settings['embedding_batch_size'],
        "chunk_batch_size": model_settings['chunk_batch_size'],
        "summary_batch_size": model_settings['summary_batch_size'],
        "chunk_size": chunk_settings['default_chunk_size'],
        "processing_chunk_size": chunk_settings['processing_chunk_size'],
        "stratification_chunk_size": chunk_settings['stratification_chunk_size']
    }

    # Add path settings based on environment
    if IS_DOCKER:
        app_prefix = "/app"
        initial_settings.update({
            "root_data_path": os.path.join(app_prefix, "data"),
            "stratified_path": os.path.join(app_prefix, "data", "stratified"),
            "knowledge_base_path": os.path.join(app_prefix, "data", "knowledge_base.csv"),
            "temp_path": os.path.join(app_prefix, "temp_files")
        })
    elif IS_REPLIT:
        initial_settings.update({
            "root_data_path": path_settings['root_data_path'],
            "stratified_path": os.path.join(path_settings['root_data_path'], 'stratified'),
            "knowledge_base_path": os.path.join(path_settings['root_data_path'], 'knowledge_base.csv'),
            "temp_path": "temp_files"
        })
    else:
        # Local development
        initial_settings.update({
            "root_data_path": path_settings['root_data_path'],
            "stratified_path": path_settings['stratified'],
            "knowledge_base_path": path_settings['knowledge_base'],
            "temp_path": path_settings['temp']
        })

    # Store settings in user session
    cl.user_session.set("settings", initial_settings)

@cl.on_settings_update
async def setup_agent(settings: dict):
    """Update chat settings."""
    try:
        # Get all configuration settings
        model_settings = Config.get_model_settings()
        processing_settings = Config.get_processing_settings()
        sample_settings = Config.get_sample_settings()
        chunk_settings = Config.get_chunk_settings()
        path_settings = Config.PATH_SETTINGS
        
        # Validate and update settings
        filter_date = settings.get("filter_date", processing_settings.get('filter_date'))
        sample_size = min(
            settings.get("sample_size", sample_settings['default_sample_size']),
            sample_settings['max_sample_size']
        )
        
        # Get provider settings with defaults
        embedding_provider = settings.get("embedding_provider", model_settings['default_embedding_provider'])
        chunk_provider = settings.get("chunk_provider", model_settings['default_chunk_provider'])
        summary_provider = settings.get("summary_provider", model_settings['default_summary_provider'])
        
        # Validate providers against ModelProvider enum values
        if any(p not in VALID_PROVIDERS for p in [embedding_provider, chunk_provider, summary_provider]):
            raise ValueError(f"Invalid provider selected. Valid providers are: {', '.join(VALID_PROVIDERS)}")

        # Store the updated settings with all necessary parameters
        updated_settings = {
            # User configurable settings
            "force_refresh": settings.get("force_refresh", False),
            "filter_date": filter_date,
            "sample_size": sample_size,
            
            # Provider settings
            "embedding_provider": embedding_provider,
            "chunk_provider": chunk_provider,
            "summary_provider": summary_provider,
            
            # Processing settings from Config
            "max_workers": processing_settings['max_workers'],
            "cache_enabled": processing_settings['cache_enabled'],
            "padding_enabled": processing_settings['padding_enabled'],
            "contraction_mapping_enabled": processing_settings['contraction_mapping_enabled'],
            "non_alpha_numeric_enabled": processing_settings['non_alpha_numeric_enabled'],
            "max_tokens": processing_settings['max_tokens'],
            
            # Batch and chunk settings from Config
            "embedding_batch_size": model_settings['embedding_batch_size'],
            "chunk_batch_size": model_settings['chunk_batch_size'],
            "summary_batch_size": model_settings['summary_batch_size'],
            "chunk_size": chunk_settings['default_chunk_size'],
            "processing_chunk_size": chunk_settings['processing_chunk_size'],
            "stratification_chunk_size": chunk_settings['stratification_chunk_size']
        }

        # Add path settings based on environment
        if IS_DOCKER:
            app_prefix = "/app"
            updated_settings.update({
                "root_data_path": os.path.join(app_prefix, "data"),
                "stratified_path": os.path.join(app_prefix, "data", "stratified"),
                "knowledge_base_path": os.path.join(app_prefix, "data", "knowledge_base.csv"),
                "temp_path": os.path.join(app_prefix, "temp_files")
            })
        elif IS_REPLIT:
            updated_settings.update({
                "root_data_path": path_settings['root_data_path'],
                "stratified_path": os.path.join(path_settings['root_data_path'], 'stratified'),
                "knowledge_base_path": os.path.join(path_settings['root_data_path'], 'knowledge_base.csv'),
                "temp_path": "temp_files"
            })
        else:
            # Local development
            updated_settings.update({
                "root_data_path": path_settings['root_data_path'],
                "stratified_path": path_settings['stratified'],
                "knowledge_base_path": path_settings['knowledge_base'],
                "temp_path": path_settings['temp']
            })

        # Update session settings
        cl.user_session.set("settings", updated_settings)

        # Log the final settings
        logger.info(f"Settings updated successfully: {updated_settings}")

        await cl.Message(
            content="✅ Settings updated successfully! Changes will be applied to your next query."
        ).send()
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        await cl.Message(content=f"❌ Failed to update settings: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    """Process user messages."""
    try:
        # Get current settings
        settings = cl.user_session.get("settings")

        # Send processing message
        processing_msg = await cl.Message(
            content="🔄 Processing your query... This may take a few minutes."
        ).send()

        try:
            # Process the query using the API
            chunks, response = await process_query(message.content, settings)

            # Create elements for chunks if available
            elements = []
            if chunks:
                # Format chunks properly
                formatted_chunks = [format_chunk(chunk) for chunk in chunks]
                elements.append(
                    cl.Text(name=f"Related Chunks ({len(chunks)} found)",
                            content="\n\n".join(formatted_chunks),
                            display="side"))

            # Send the response with a header indicating the number of chunks
            header = f"📊 Analysis based on {len(chunks)} relevant chunks:\n\n"
            await cl.Message(content=header + response, elements=elements).send()

            # Remove the processing message
            if isinstance(processing_msg, cl.Message):
                await processing_msg.remove()

        except Exception as e:
            logger.error(f"Error in knowledge processing: {str(e)}")
            logger.error(f"Full error details: {str(e)}", exc_info=True)
            await cl.Message(content=f"❌ {str(e)}").send()
            if isinstance(processing_msg, cl.Message):
                await processing_msg.remove()

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        await cl.Message(content=f"❌ {str(e)}").send()
    except Exception as e:
        logger.error(f"Error in message handling: {str(e)}")
        logger.error(f"Full error details: {str(e)}", exc_info=True)
        await cl.Message(
            content=f"❌ An error occurred while processing your query: {str(e)}"
        ).send()

if __name__ == "__main__":
    import platform
    import os

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    from chainlit.cli import run_chainlit
    app_path = os.path.abspath(__file__)
    logger.info(f"Starting Chainlit with app path: {app_path}")
    run_chainlit(app_path, host='0.0.0.0', port=8000)