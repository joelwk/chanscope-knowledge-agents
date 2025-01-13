import chainlit as cl
import logging
import os
from chainlit.input_widget import Select, Switch, Slider, TextInput
import aiohttp
from aiohttp import ClientTimeout
import asyncio
from chainlit_frontend import API_BASE_URL, is_docker
from datetime import datetime
from config.settings import Config

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
API_TIMEOUT = ClientTimeout(
    total=600, connect=120)  # Increase total timeout to 10 minutes

async def process_query(query: str, settings: dict):
    """Process a query using the API endpoint."""
    last_error = None

    # Get all possible base URLs to try
    api_settings = Config.get_api_settings()
    base_urls = api_settings.get('base_urls', [])

    # In Replit, ensure we use the correct URL format
    if os.getenv('REPL_SLUG') and os.getenv('REPL_OWNER'):
        replit_url = f"https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co"
        if replit_url not in base_urls:
            base_urls.insert(0, replit_url)  # Add Replit URL as primary

        # Adjust batch size for Replit environment
        original_batch_size = settings.get('batch_size', 100)
        settings['batch_size'] = min(original_batch_size, 100)
        if settings['batch_size'] != original_batch_size:
            logger.warning(f"Reduced batch size from {original_batch_size} to {settings['batch_size']} for Replit environment")

    logger.info("=== API Connection Attempt ===")
    logger.info(f"Available API endpoints: {base_urls}")
    logger.info(f"Using settings: {settings}")

    # Create session with proper connection settings
    connector = aiohttp.TCPConnector(
        force_close=False,  # Allow connection reuse
        keepalive_timeout=300,  # 5 minutes keepalive
        enable_cleanup_closed=True  # Clean up closed connections
    )

    async with aiohttp.ClientSession(
        timeout=API_TIMEOUT,
        connector=connector
    ) as session:
        logger.info("Created aiohttp session")

        # Try each base URL in order
        for base_url in base_urls:
            try:
                logger.info(f"Attempting to connect to API at {base_url}")
                logger.info(f"Full request URL: {base_url}/process_query")
                logger.info(f"Request payload: {{'query': {query}, **{settings}}}")

                async with session.post(
                    f"{base_url}/process_query",
                    json={
                        "query": query,
                        **settings},
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"}
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
                last_error = e
                continue

    # If we get here, all URLs failed
    logger.error(f"All API endpoints failed. Last error: {str(last_error)}")
    logger.error(f"Connection details: host={api_settings['host']}, port={api_settings['port']}")
    raise Exception(f"Failed to connect to any API endpoint. Please try again in a moment.")

def format_chunk(chunk):
    """Format a chunk for display."""
    if isinstance(chunk, dict):
        return f"Score: {chunk.get('score', 'N/A')}\nContent: {chunk.get('content', 'No content available')}"
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

    # Get settings from Config
    processing_settings = Config.get_processing_settings()
    provider_settings = Config.get_provider_settings()

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
        Slider(id="batch_size",
               label="Batch Size",
               initial=processing_settings['batch_size'],
               min=10,
               max=500,
               step=10,
               description="Number of items to process in each batch"),
        # Model provider settings
        Select(id="embedding_provider",
               label="Embedding Provider",
               values=["openai", "grok", "venice"],
               initial_index=["openai", "grok", "venice"
                              ].index(provider_settings['embedding_provider']),
               description="Select the provider for embeddings"),
        Select(id="chunk_provider",
               label="Chunk Provider",
               values=["openai", "grok", "venice"],
               initial_index=["openai", "grok", "venice"
                              ].index(provider_settings['chunk_provider']),
               description="Select the provider for text chunking"),
        Select(id="summary_provider",
               label="Summary Provider",
               values=["openai", "grok", "venice"],
               initial_index=["openai", "grok", "venice"
                              ].index(provider_settings['summary_provider']),
               description="Select the provider for summarization"),
    ]).send()

    # Store initial settings
    cl.user_session.set(
        "settings", {
            "force_refresh": False,
            "filter_date": processing_settings['filter_date'],
            "batch_size": processing_settings['batch_size'],
            "max_workers": processing_settings['max_workers'],
            "embedding_provider": provider_settings['embedding_provider'],
            "chunk_provider": provider_settings['chunk_provider'],
            "summary_provider": provider_settings['summary_provider']
        })


@cl.on_settings_update
async def setup_agent(settings: dict):
    """Update chat settings."""
    try:
        # Log settings update
        logger.info(f"Updating settings: {settings}")

        # Validate date format
        filter_date = settings.get("filter_date")
        try:
            if filter_date:
                datetime.strptime(filter_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(
                "Invalid date format. Please use YYYY-MM-DD format.")

        # Store the updated settings
        cl.user_session.set(
            "settings", {
                "force_refresh":
                settings.get("force_refresh", False),
                "filter_date":
                filter_date,
                "batch_size":
                int(settings.get("batch_size", Config.DEFAULT_BATCH_SIZE)),
                "max_workers":
                None,
                "embedding_provider":
                settings.get("embedding_provider",
                             Config.DEFAULT_EMBEDDING_PROVIDER),
                "chunk_provider":
                settings.get("chunk_provider", Config.DEFAULT_CHUNK_PROVIDER),
                "summary_provider":
                settings.get("summary_provider",
                             Config.DEFAULT_SUMMARY_PROVIDER)
            })

        await cl.Message(
            content=
            "✅ Settings updated successfully! Changes will be applied to your next query."
        ).send()
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        await cl.Message(content=f"❌ Failed to update settings: {str(e)}"
                         ).send()


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
                    cl.Text(name="Related Chunks",
                            content="\n\n---\n\n".join(formatted_chunks),
                            display="side"))

            # Send the response
            await cl.Message(content=response, elements=elements).send()

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
    run_chainlit(app_path, host='0.0.0.0', port=5000)