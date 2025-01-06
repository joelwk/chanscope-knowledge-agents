import chainlit as cl
import logging
import os
from chainlit.input_widget import Select, Switch, Slider, TextInput
import aiohttp
from aiohttp import ClientTimeout
import asyncio
from chainlit_frontend import API_BASE_URL, is_docker, api_host, api_port
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# API timeout configuration - increased to 20 minutes
API_TIMEOUT = ClientTimeout(total=1500, connect=60)  # 1200 seconds = 20 minutes

async def process_query(query: str, settings: dict):
    """Process a query using the API endpoint."""
    try:
        logger.info(f"Attempting to connect to API at {API_BASE_URL}")
        logger.info(f"Docker environment: {is_docker}")
        logger.info(f"Full request URL: {API_BASE_URL}/process_query")
        logger.info(f"Request payload: {{'query': {query}, **{settings}}}")
        
        # Create session with keep-alive
        async with aiohttp.ClientSession(timeout=API_TIMEOUT, 
                                       connector=aiohttp.TCPConnector(keepalive_timeout=300)) as session:
            logger.info("Created aiohttp session")
            try:
                async with session.post(
                    f"{API_BASE_URL}/process_query",
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
            except asyncio.TimeoutError:
                logger.error("API request timed out")
                raise Exception(
                    "The request is taking longer than expected. The API is still processing your request. "
                    "Please wait or try again with a simpler query or smaller batch size."
                )
    except aiohttp.ClientError as e:
        logger.error(f"API connection error: {str(e)}")
        logger.error(f"Connection details: host={api_host}, port={api_port}, is_docker={is_docker}")
        raise Exception(f"Failed to connect to the API service at {API_BASE_URL}. Please try again in a moment.")

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
        content="Welcome to Chanscope. Before you start, please review the Readme to understand how to use this tool. It is not a chatbot, but a tool for querying 4chan data to generate insights and forecasts."
    ).send()

    # Initialize chat settings with all configuration options
    settings = await cl.ChatSettings(
        [
            # Processing settings
            Switch(
                id="process_new",
                label="Process New Data",
                initial=False,
                description="Enable to process new data in the knowledge base"
            ),
            TextInput(
                id="filter_date",
                label="Filter Date (YYYY-MM-DD)",
                initial=os.getenv('FILTER_DATE', '2024-12-24'),
                description="Date to filter data from (format: YYYY-MM-DD)"
            ),
            Slider(
                id="batch_size",
                label="Batch Size",
                initial=int(os.getenv('BATCH_SIZE', 100)),
                min=10,
                max=500,
                step=10,
                description="Number of items to process in each batch"
            ),
            # Model provider settings
            Select(
                id="embedding_provider",
                label="Embedding Provider",
                values=["openai", "grok", "venice"],
                initial_index=["openai", "grok", "venice"].index(os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai')),
                description="Select the provider for embeddings"
            ),
            Select(
                id="chunk_provider",
                label="Chunk Provider",
                values=["openai", "grok", "venice"],
                initial_index=["openai", "grok", "venice"].index(os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai')),
                description="Select the provider for text chunking"
            ),
            Select(
                id="summary_provider",
                label="Summary Provider",
                values=["openai", "grok", "venice"],
                initial_index=["openai", "grok", "venice"].index(os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai')),
                description="Select the provider for summarization"
            ),
        ]
    ).send()

    # Store initial settings
    cl.user_session.set("settings", {
        "process_new": False,
        "filter_date": os.getenv('FILTER_DATE', '2024-12-24'),
        "batch_size": int(os.getenv('BATCH_SIZE', 100)),
        "max_workers": None,
        "embedding_provider": os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai'),
        "chunk_provider": os.getenv('DEFAULT_CHUNK_PROVIDER', 'openai'),
        "summary_provider": os.getenv('DEFAULT_SUMMARY_PROVIDER', 'openai')
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
            raise ValueError("Invalid date format. Please use YYYY-MM-DD format.")
        
        # Store the updated settings
        cl.user_session.set("settings", {
            "process_new": settings.get("process_new", False),
            "filter_date": filter_date,
            "batch_size": int(settings.get("batch_size", 100)),
            "max_workers": None,
            "embedding_provider": settings.get("embedding_provider", "openai"),
            "chunk_provider": settings.get("chunk_provider", "openai"),
            "summary_provider": settings.get("summary_provider", "openai")
        })

        await cl.Message(
            content="✅ Settings updated successfully! Changes will be applied to your next query."
        ).send()
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        await cl.Message(
            content=f"❌ Failed to update settings: {str(e)}"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Process user messages."""
    try:
        # Get current settings
        settings = cl.user_session.get("settings")
        
        # Send processing message
        processing_msg = await cl.Message(content="🔄 Processing your query... This may take a few minutes.").send()
        
        try:
            # Process the query using the API
            chunks, response = await process_query(message.content, settings)
            
            # Create elements for chunks if available
            elements = []
            if chunks:
                # Format chunks properly
                formatted_chunks = [format_chunk(chunk) for chunk in chunks]
                elements.append(
                    cl.Text(
                        name="Related Chunks",
                        content="\n\n---\n\n".join(formatted_chunks),
                        display="side"
                    )
                )

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
        await cl.Message(
            content=f"❌ {str(e)}"
        ).send()
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
        # Set the event loop policy for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    from chainlit.cli import run_chainlit
    # Use the absolute path of the current file
    app_path = os.path.abspath(__file__)
    logger.info(f"Starting Chainlit with app path: {app_path}")
    run_chainlit(app_path) 