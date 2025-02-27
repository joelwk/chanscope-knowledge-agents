"""Model operations and API interactions module."""
import os
import yaml
import logging
import asyncio
import threading
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from openai import AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pathlib import Path
from config.base_settings import get_base_settings
from config.settings import Config
import tiktoken
import hashlib
import numpy as np
# Initialize logging
logger = logging.getLogger(__name__)

# Thread-safe singleton pattern
_instance = None
_instance_lock = threading.Lock()

class ModelProviderError(Exception):
    """Base exception for model provider errors."""
    pass

class ModelConfigurationError(Exception):
    """Exception for model configuration errors."""
    pass

class ModelOperationError(Exception):
    """Exception for model operation errors."""
    pass

class EmbeddingError(ModelOperationError):
    """Exception for embedding-specific errors."""
    pass

class ChunkGenerationError(ModelOperationError):
    """Exception for chunk generation errors."""
    pass

class SummarizationError(ModelOperationError):
    """Exception for summarization errors."""
    pass

class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    GROK = "grok"
    VENICE = "venice"

    @classmethod
    def from_str(cls, value: str) -> 'ModelProvider':
        """Convert string to ModelProvider enum, handling carriage returns."""
        if value is None:
            raise ValueError("Provider value cannot be None")
        # Clean the value of any carriage returns or whitespace
        cleaned_value = value.strip().lower()
        try:
            return cls(cleaned_value)
        except ValueError:
            raise ValueError(f"'{value}' is not a valid ModelProvider")

    def __str__(self) -> str:
        """Return the string value of the provider."""
        return self.value

# Initialize client locks after ModelProvider is defined
_client_locks = {
    ModelProvider.OPENAI: asyncio.Lock(),
    ModelProvider.GROK: asyncio.Lock(),
    ModelProvider.VENICE: asyncio.Lock()
}

class ModelOperation(str, Enum):
    """Types of model operations."""
    EMBEDDING = "embedding"
    CHUNK_GENERATION = "chunk_generation"
    SUMMARIZATION = "summarization"

class ModelConfig:
    """Model configuration with validated settings."""
    def __init__(self, **settings):
        """Initialize ModelConfig with validated settings."""
        if not settings:
            settings = get_base_settings()

        # Get model settings
        model_settings = settings.get('model', {})

        # Model settings
        self.embedding_batch_size = model_settings.get('embedding_batch_size', 10)
        self.chunk_batch_size = model_settings.get('chunk_batch_size', 5)
        self.summary_batch_size = model_settings.get('summary_batch_size', 3)
        self.default_embedding_provider = model_settings.get('default_embedding_provider')
        self.default_chunk_provider = model_settings.get('default_chunk_provider')
        self.default_summary_provider = model_settings.get('default_summary_provider')

        # Get path settings
        path_settings = settings.get('paths', {})

        # Path settings - convert to strings
        self.root_data_path = str(path_settings.get('root_data_path', 'data'))
        self.stratified_path = str(path_settings.get('stratified', 'data/stratified'))
        self.temp_path = str(path_settings.get('temp', 'temp_files'))

        # Get processing settings
        self.processing_settings = settings.get('processing', {})

        # Store path settings for easy access
        self.path_settings = {
            'root_data_path': self.root_data_path,
            'stratified': self.stratified_path,
            'temp': self.temp_path
        }

        # Store model settings for easy access
        self.model_settings = model_settings

        # Extract common processing settings for direct access
        self.filter_date = self.processing_settings.get('filter_date')
        self.sample_size = self.processing_settings.get('sample_size', 1500)
        self.max_workers = self.processing_settings.get('max_workers', 4)

    @property
    def paths(self) -> Dict[str, str]:
        """Get path settings."""
        return self.path_settings

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create ModelConfig from environment settings."""
        return cls()

    def get_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the configured provider for a given operation."""
        provider_map = {
            ModelOperation.EMBEDDING: self.default_embedding_provider,
            ModelOperation.CHUNK_GENERATION: self.default_chunk_provider,
            ModelOperation.SUMMARIZATION: self.default_summary_provider
        }
        return ModelProvider(provider_map[operation])

    def get_batch_size(self, operation: ModelOperation) -> int:
        """Get the configured batch size for a given operation."""
        batch_map = {
            ModelOperation.EMBEDDING: self.embedding_batch_size,
            ModelOperation.CHUNK_GENERATION: self.chunk_batch_size,
            ModelOperation.SUMMARIZATION: self.summary_batch_size
        }
        return batch_map[operation]

def _get_config_hash(config: ModelConfig) -> str:
    """Generate a hash of configuration settings that affect the data processing.

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
            'root_data_path': config.root_data_path,
            'stratified': config.stratified_path
        }
    }

    # Create hash
    settings_str = json.dumps(settings, sort_keys=True)
    return hashlib.md5(settings_str.encode()).hexdigest()

class EmbeddingResponse(BaseModel):
    """Standardized embedding response across providers."""
    embedding: Union[List[float], List[List[float]]]
    model: str
    usage: Dict[str, int]

def load_prompts(prompt_path: str = None) -> Dict[str, Any]:
    """Load prompts from YAML file."""
    try:
        if prompt_path is None:
            # Get the path to the knowledge_agents directory
            current_dir = Path(__file__).parent
            prompt_path = current_dir / 'prompt.yaml'
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
        with open(prompt_path, 'r') as file:
            prompts = yaml.safe_load(file)
        if not isinstance(prompts, dict):
            raise ValueError(f"Invalid prompt file format. Expected dict, got {type(prompts)}")
        required_sections = {"system_prompts", "user_prompts"}
        missing_sections = required_sections - set(prompts.keys())
        if missing_sections:
            raise ValueError(f"Missing required sections in prompts: {missing_sections}")
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts from {prompt_path}: {str(e)}")
        raise

def load_config() -> ModelConfig:
    """Load configuration using BaseConfig.

    Returns:
        ModelConfig: Configuration instance with validated settings
    """
    if not hasattr(load_config, '_config_cache'):
        try:
            # Create model config directly from BaseConfig
            model_config = ModelConfig()

            # Validate required paths exist
            for path_name, path in model_config.paths.items():
                path_obj = Path(path)
                if path_name != 'temp':  # temp directory is created on demand
                    if not path_obj.exists():
                        logger.warning(f"Creating required directory: {path}")
                        path_obj.mkdir(parents=True, exist_ok=True)

            # Cache the configuration
            load_config._config_cache = model_config

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise ModelConfigurationError(f"Failed to load configuration: {str(e)}") from e

    return load_config._config_cache

class KnowledgeAgent:
    """Thread-safe singleton class for model operations."""

    def __new__(cls):
        """Ensure singleton pattern with thread safety."""
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:
                    _instance = super(KnowledgeAgent, cls).__new__(cls)
                    _instance._initialized = False
        return _instance

    def __init__(self):
        """Initialize the agent if not already initialized."""
        if not getattr(self, '_initialized', False):
            with _instance_lock:
                if not getattr(self, '_initialized', False):
                    self._config = None
                    self._chunk_lock = asyncio.Lock()
                    self._embedding_lock = asyncio.Lock()
                    self.prompts = load_prompts()
                    self._clients = self._initialize_clients()
                    self._initialized = True
                    logger.info("Initialized KnowledgeAgent singleton")

    async def _get_client(self, provider: ModelProvider) -> Any:
        """Get or create a client for the specified provider with proper locking."""
        async with _client_locks[provider]:
            if provider not in self._clients:
                self._clients[provider] = await self._create_client(provider)
            return self._clients[provider]

    async def _create_client(self, provider: ModelProvider) -> Any:
        """Create a new client for the specified provider."""
        try:
            # Import Config here to avoid circular dependency
            from config.settings import Config

            if provider == ModelProvider.OPENAI:
                return AsyncOpenAI(api_key=Config.get_openai_api_key())
            elif provider == ModelProvider.GROK:
                grok_key = Config.get_grok_api_key()
                if not grok_key:
                    raise ModelProviderError("Grok API key not found")
                return AsyncOpenAI(
                    api_key=grok_key,
                    base_url=get_base_settings()['model'].get('grok_api_base')
                )
            elif provider == ModelProvider.VENICE:
                venice_key = Config.get_venice_api_key()
                if not venice_key:
                    raise ModelProviderError("Venice API key not found")
                return AsyncOpenAI(
                    api_key=venice_key,
                    base_url=get_base_settings()['model'].get('venice_api_base')
                )
            else:
                raise ModelProviderError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Error creating client for provider {provider}: {str(e)}")
            logger.error(f"Provider settings: {get_base_settings()['model']}")
            raise ModelProviderError(f"Failed to create client for {provider}: {str(e)}")

    def _validate_config(self, config: Optional[ModelConfig] = None) -> ModelConfig:
        """Validate and return configuration."""
        if config is None:
            if self._config is None:
                self._config = ModelConfig.from_env()
            return self._config
        return config

    def _initialize_clients(self) -> Dict[str, AsyncOpenAI]:
        """Initialize model clients dynamically based on provided configuration."""
        clients = {}
        base_settings = get_base_settings()

        # Initialize OpenAI client
        openai_api_key = Config.get_openai_api_key()
        if not openai_api_key:
            logger.warning("OpenAI API key is missing or empty")
        else:
            try:
                clients[ModelProvider.OPENAI.value] = AsyncOpenAI(
                    api_key=openai_api_key,
                    base_url=base_settings['model'].get('openai_api_base', 'https://api.openai.com/v1'),
                    max_retries=5
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise ModelConfigurationError(f"Failed to initialize OpenAI client: {str(e)}") from e

        # Initialize Grok client using OpenAI framework
        grok_api_key = Config.get_grok_api_key()
        if grok_api_key:
            try:
                clients[ModelProvider.GROK.value] = AsyncOpenAI(
                    api_key=grok_api_key,
                    base_url=base_settings['model'].get('grok_api_base'),
                    max_retries=5
                )
                logger.info("Grok client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Grok client: {str(e)}")

        # Initialize Venice client using OpenAI framework
        venice_api_key = Config.get_venice_api_key()
        if venice_api_key:
            try:
                clients[ModelProvider.VENICE.value] = AsyncOpenAI(
                    api_key=venice_api_key,
                    base_url=base_settings['model'].get('venice_api_base'),
                    max_retries=5
                )
                logger.info("Venice client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Venice client: {str(e)}")

        if not clients:
            raise ModelConfigurationError("No API providers configured. Please set at least one of the following in settings:\n"
                           "- OPENAI_API_KEY (Required)\n"
                           "- GROK_API_KEY (Optional)\n"
                           "- VENICE_API_KEY (Optional)")
        return clients

    def _get_model_name(self, provider: ModelProvider, operation: ModelOperation) -> str:
        """Get the appropriate model name for a provider and operation type."""
        try:
            # Import Config here to avoid circular dependency
            from config.settings import Config

            if operation == ModelOperation.EMBEDDING:
                if provider == ModelProvider.OPENAI:
                    return Config.get_openai_embedding_model()
                elif provider == ModelProvider.GROK:
                    return Config.get_grok_model() or "grok-v1-embedding"
                else:
                    raise ModelProviderError(f"Unsupported embedding provider: {provider}")
            elif operation == ModelOperation.CHUNK_GENERATION:
                if provider == ModelProvider.OPENAI:
                    return Config.get_openai_model()
                elif provider == ModelProvider.GROK:
                    return Config.get_grok_model()
                elif provider == ModelProvider.VENICE:
                    return Config.get_venice_chunk_model()
                else:
                    raise ModelProviderError(f"Unsupported chunk generation provider: {provider}")
            elif operation == ModelOperation.SUMMARIZATION:
                if provider == ModelProvider.OPENAI:
                    return Config.get_openai_model()
                elif provider == ModelProvider.GROK:
                    return Config.get_grok_model()
                elif provider == ModelProvider.VENICE:
                    return Config.get_venice_model()
                else:
                    raise ModelProviderError(f"Unsupported summarization provider: {provider}")
            else:
                raise ModelOperationError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error getting model name for {provider} and operation {operation}: {str(e)}")
            raise ModelProviderError(f"Failed to get model name: {str(e)}")

    async def generate_summary(
        self, 
        query: str, 
        results: str, 
        context: Optional[str] = None,
        temporal_context: Optional[Dict[str, str]] = None,
        provider: Optional[ModelProvider] = None
    ) -> str:
        """Generate a summary using the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.SUMMARIZATION)

        try:
            client = await self._get_client(provider)
            model = self._get_model_name(provider, ModelOperation.SUMMARIZATION)

            # Ensure temporal context is properly formatted
            if temporal_context is None:
                temporal_context = {
                    "start_date": "Unknown",
                    "end_date": "Unknown"
                }

            # Create base message content without results first
            base_content = self.prompts["user_prompts"]["summary_generation"]["content"].format(
                query=query,
                temporal_context=f"Time Range: {temporal_context['start_date']} to {temporal_context['end_date']}",
                context=context or "No additional context provided.",
                results="",  # Placeholder
                start_date=temporal_context['start_date'],
                end_date=temporal_context['end_date']
            )

            system_content = self.prompts["system_prompts"]["objective_analysis"]["content"]

            # Calculate tokens for base content
            encoding = tiktoken.get_encoding("cl100k_base")
            base_tokens = len(encoding.encode(base_content)) + len(encoding.encode(system_content))

            # Available tokens for results (leaving room for response)
            available_tokens = 7000 - base_tokens

            # Parse results and create chunks if needed
            try:
                chunk_results = json.loads(results)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing results JSON: {str(e)}")
                raise SummarizationError("Invalid results format") from e

            if not isinstance(chunk_results, list):
                chunk_results = [chunk_results]

            # Convert chunks to text format for token counting
            chunks_text = json.dumps(chunk_results, indent=2)
            chunks_tokens = len(encoding.encode(chunks_text))

            if chunks_tokens > available_tokens:
                logger.warning(f"Results exceed token limit ({chunks_tokens} > {available_tokens}). Truncating...")
                # Use create_chunks to manage token size
                from .inference_ops import create_chunks
                chunks = create_chunks(chunks_text, available_tokens, encoding)

                if chunks:
                    chunks_text = chunks[0]["text"]
                    logger.info(f"Truncated results to {len(encoding.encode(chunks_text))} tokens")
                else:
                    logger.warning("Fallback to simple truncation")
                    # If chunking fails, take a simple truncation approach
                    truncated_results = []
                    current_tokens = 0
                    for chunk in chunk_results:
                        chunk_str = json.dumps({
                            "thread_id": chunk["thread_id"],
                            "posted_date_time": chunk["posted_date_time"],
                            "analysis": {
                                "thread_analysis": chunk["analysis"]["thread_analysis"][:500],
                                "metrics": chunk["analysis"]["metrics"]
                            }
                        })
                        chunk_tokens = len(encoding.encode(chunk_str))
                        if current_tokens + chunk_tokens < available_tokens:
                            truncated_results.append(json.loads(chunk_str))
                            current_tokens += chunk_tokens
                        else:
                            break
                    chunks_text = json.dumps(truncated_results, indent=2)

            # Create final messages with properly sized results
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": self.prompts["user_prompts"]["summary_generation"]["content"].format(
                        query=query,
                        temporal_context=f"Time Range: {temporal_context['start_date']} to {temporal_context['end_date']}",
                        context=context or "No additional context provided.",
                        results=chunks_text,
                        start_date=temporal_context['start_date'],
                        end_date=temporal_context['end_date']
                    )
                }
            ]

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                presence_penalty=0.2,
                frequency_penalty=0.2
            )
            return response.choices[0].message.content

        except SummarizationError:
            raise
        except Exception as e:
            logger.error(f"Error generating summary with {provider}: {str(e)}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self._clients:
                logger.warning(f"Falling back to OpenAI for summary generation")
                return await self.generate_summary(
                    query, 
                    results, 
                    context=context,
                    temporal_context=temporal_context,
                    provider=ModelProvider.OPENAI
                )
            raise SummarizationError(f"Failed to generate summary: {str(e)}") from e

    async def generate_chunks(
        self, 
        content: str,
        provider: Optional[ModelProvider] = None
    ) -> Dict[str, str]:
        """Generate chunks using the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.CHUNK_GENERATION)

        client = await self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.CHUNK_GENERATION)

        async with self._chunk_lock:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.prompts["system_prompts"]["generate_chunks"]["content"]
                        },
                        {
                            "role": "user",
                            "content": self.prompts["user_prompts"]["text_chunk_summary"]["content"].format(
                                content=content
                            )
                        }
                    ],
                    temperature=0.1,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )

                result = response.choices[0].message.content
                if not result:
                    raise ChunkGenerationError("Empty response from model")

                # Split on signal_context as per prompt template
                sections = result.split("<signal_context>")

                if len(sections) > 1:
                    thread_analysis = sections[0].strip()
                    signal_context = sections[1].strip()

                    # Parse thread analysis metrics
                    metrics = {}
                    for line in thread_analysis.split('\n'):
                        if '(' in line and ')' in line:
                            try:
                                # Extract metrics from format: (t, metric, value, confidence)
                                metric_str = line[line.find("(")+1:line.find(")")].strip()
                                parts = [p.strip().strip("'\"") for p in metric_str.split(',')]
                                if parts[0] == 't' or 'metric' in parts:
                                    continue
                                if len(parts) >= 4:
                                    timestamp, metric_name, value, confidence = parts[:4]
                                    try:
                                        metrics[metric_name] = float(value)
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"Could not convert metric value to float: {value}")
                            except Exception as e:
                                logger.warning(f"Failed to parse metric line: {line}, error: {e}")

                    # Parse context sections
                    context_elements = self._parse_context_sections(signal_context)

                    return {
                        "analysis": {
                            "thread_analysis": thread_analysis,
                            "metrics": metrics
                        },
                        "context": context_elements,
                        "metrics": {
                            "sections": len(sections),
                            "analysis_length": len(thread_analysis),
                            "context_length": len(signal_context),
                            **metrics
                        }
                    }
                else:
                    return {
                        "analysis": {
                            "thread_analysis": result.strip(),
                            "metrics": {}
                        },
                        "context": {
                            "key_claims": [],
                            "supporting_text": [],
                            "risk_assessment": [],
                            "viral_potential": []
                        },
                        "metrics": {
                            "sections": 1,
                            "analysis_length": len(result),
                            "context_length": 0
                        }
                    }
            except ChunkGenerationError:
                raise
            except Exception as e:
                logger.error(f"Error generating chunks with {provider}: {str(e)}")
                if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self._clients:
                    logger.warning(f"Falling back to OpenAI for chunk generation")
                    return await self.generate_chunks(content, provider=ModelProvider.OPENAI)
                raise ChunkGenerationError(f"Failed to generate chunks: {str(e)}") from e

    async def generate_chunks_batch(
        self,
        contents: List[str],
        provider: Optional[ModelProvider] = None,
        chunk_batch_size: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Generate chunks for multiple contents in batches."""
        config = self._validate_config()
        provider = provider or self._get_default_provider(ModelOperation.CHUNK_GENERATION)
        chunk_batch_size = chunk_batch_size or config.chunk_batch_size

        results = []
        for i in range(0, len(contents), chunk_batch_size):
            batch = contents[i:i + chunk_batch_size]
            batch_results = await asyncio.gather(
                *[self.generate_chunks(content, provider=provider) for content in batch]
            )
            results.extend(batch_results)
        return results

    async def generate_summaries_batch(
        self,
        queries: List[str],
        results_list: List[str],
        contexts: Optional[List[str]] = None,
        temporal_contexts: Optional[List[Dict[str, str]]] = None,
        provider: Optional[ModelProvider] = None,
        summary_batch_size: Optional[int] = None
    ) -> List[str]:
        """Generate summaries for multiple queries in batches."""
        config = self._validate_config()
        provider = provider or self._get_default_provider(ModelOperation.SUMMARIZATION)
        summary_batch_size = summary_batch_size or config.summary_batch_size

        if contexts is None:
            contexts = [None] * len(queries)
        if temporal_contexts is None:
            temporal_contexts = [None] * len(queries)

        results = []
        for i in range(0, len(queries), summary_batch_size):
            batch_queries = queries[i:i + summary_batch_size]
            batch_results = results_list[i:i + summary_batch_size]
            batch_contexts = contexts[i:i + summary_batch_size]
            batch_temporal = temporal_contexts[i:i + summary_batch_size]

            batch_summaries = await asyncio.gather(
                *[self.generate_summary(
                    query=query,
                    results=result,
                    context=context,
                    temporal_context=temporal,
                    provider=provider
                ) for query, result, context, temporal in zip(
                    batch_queries, batch_results, batch_contexts, batch_temporal
                )]
            )
            results.extend(batch_summaries)
        return results

    def _process_chunk_response(self, result: str) -> Dict[str, Any]:
        """Process chunk response and extract structured data."""
        if not result:
            raise ValueError("Empty response from model")

        # Split on signal_context as per prompt template
        sections = result.split("<signal_context>")

        if len(sections) > 1:
            thread_analysis = sections[0].strip()
            signal_context = sections[1].strip()

            # Parse thread analysis metrics
            metrics = {}
            for line in thread_analysis.split('\n'):
                if '(' in line and ')' in line:
                    try:
                        metric_str = line[line.find("(")+1:line.find(")")].strip()
                        parts = [p.strip().strip("'\"") for p in metric_str.split(',')]
                        if parts[0] == 't' or 'metric' in parts:
                            continue
                        if len(parts) >= 4:
                            timestamp, metric_name, value, confidence = parts[:4]
                            try:
                                metrics[metric_name] = float(value)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert metric value to float: {value}")
                    except Exception as e:
                        logger.warning(f"Failed to parse metric line: {line}, error: {e}")

            # Parse context sections
            context_elements = self._parse_context_sections(signal_context)

            return {
                "analysis": {
                    "thread_analysis": thread_analysis,
                    "metrics": metrics
                },
                "context": context_elements,
                "metrics": {
                    "sections": len(sections),
                    "analysis_length": len(thread_analysis),
                    "context_length": len(signal_context),
                    **metrics
                }
            }
        else:
            return {
                "analysis": {
                    "thread_analysis": result.strip(),
                    "metrics": {}
                },
                "context": {
                    "key_claims": [],
                    "supporting_text": [],
                    "risk_assessment": [],
                    "viral_potential": []
                },
                "metrics": {
                    "sections": 1,
                    "analysis_length": len(result),
                    "context_length": 0
                }
            }

    def _parse_context_sections(self, signal_context: str) -> Dict[str, List[str]]:
        """Parse context sections from signal context."""
        context_elements = {
            "key_claims": [],
            "supporting_text": [],
            "risk_assessment": [],
            "viral_potential": []
        }

        current_section = None
        for line in signal_context.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            lower_line = line.lower()
            if "key claims" in lower_line:
                current_section = "key_claims"
            elif "supporting" in lower_line and "text" in lower_line:
                current_section = "supporting_text"
            elif "risk" in lower_line and "assessment" in lower_line:
                current_section = "risk_assessment"
            elif "viral" in lower_line and "potential" in lower_line:
                current_section = "viral_potential"
            elif current_section and (line.startswith('-') or line.startswith('*')):
                content = line.lstrip('-* ').strip()
                if content:
                    context_elements[current_section].append(content)

        return context_elements

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
    async def embedding_request(
        self,
        text: Union[str, List[str]],
        provider: Optional[ModelProvider] = None,
        batch_size: Optional[int] = None
    ) -> EmbeddingResponse:
        """Request embeddings from the specified provider with batching support."""
        async with self._embedding_lock:
            config = self._validate_config()
            provider = provider or self._get_default_provider(ModelOperation.EMBEDDING)
            batch_size = batch_size or config.embedding_batch_size

            # Generate a deterministic hash for mock embeddings based on text content
            def generate_mock_embeddings(input_texts: List[str], dim: int = 3072) -> List[List[float]]:
                """Generate mock embeddings for when no API provider is available."""
                import hashlib
                import numpy as np

                mock_embeddings = []
                for t in input_texts:
                    # Create a deterministic seed from the text hash
                    seed = int(hashlib.md5(t.encode('utf-8')).hexdigest(), 16) % (2**32)
                    np.random.seed(seed)

                    # Generate a normalized random embedding
                    mock_embedding = np.random.normal(0, 0.1, dim)
                    mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
                    mock_embeddings.append(mock_embedding.tolist())

                return mock_embeddings

            # Ensure text is properly formatted
            if isinstance(text, str):
                input_text = text.strip()
                if not input_text:
                    raise ValueError("Empty text input")
                texts_to_process = [input_text]
            elif isinstance(text, list):
                # Filter out empty strings and non-strings
                texts_to_process = [t.strip() for t in text if isinstance(t, str) and t.strip()]
                if not texts_to_process:
                    raise ValueError("No valid text inputs provided")
            else:
                raise ValueError(f"Unsupported text input type: {type(text)}")

            # Check if the provider is configured
            try:
                # If provider is not in clients or the value is None, generate mock embeddings
                if provider.value not in self._clients or self._clients.get(provider.value) is None:
                    logger.warning(f"Provider {provider} not configured, generating mock embeddings")
                    mock_embeddings = generate_mock_embeddings(texts_to_process)
                    return EmbeddingResponse(
                        embedding=mock_embeddings if len(mock_embeddings) > 1 else mock_embeddings[0],
                        model="mock-embedding-model",
                        usage={"prompt_tokens": 0, "total_tokens": 0}
                    )

                # Continue with normal flow if provider is configured
                client = await self._get_client(provider)
                model = self._get_model_name(provider, ModelOperation.EMBEDDING)
            except Exception as e:
                logger.warning(f"Failed to get client for provider {provider}: {str(e)}")
                # Generate mock embeddings as fallback
                mock_embeddings = generate_mock_embeddings(texts_to_process)
                return EmbeddingResponse(
                    embedding=mock_embeddings if len(mock_embeddings) > 1 else mock_embeddings[0],
                    model="mock-embedding-model",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )

            try:
                # Get token counts for validation with robust tiktoken handling
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to get cl100k_base encoding: {e}, falling back to p50k_base")
                    try:
                        encoding = tiktoken.get_encoding("p50k_base")
                    except Exception as e2:
                        logger.error(f"Failed to get fallback encoding: {e2}")
                        # If we can't get any encoding, use a simple length-based estimate
                        token_counts = [len(t.split()) * 1.3 for t in texts_to_process]  # Rough estimate
                    else:
                        token_counts = [len(encoding.encode(t)) for t in texts_to_process]
                else:
                    token_counts = [len(encoding.encode(t)) for t in texts_to_process]

                # Validate token counts (OpenAI limit is 8191 per text)
                MAX_TOKENS_PER_TEXT = 8191
                for text_item, token_count in zip(texts_to_process, token_counts):
                    if token_count > MAX_TOKENS_PER_TEXT:
                        logger.warning(f"Text exceeds token limit ({token_count} > {MAX_TOKENS_PER_TEXT}). Truncating...")
                        # If we have encoding, use it for truncation
                        if 'encoding' in locals():
                            truncated_text = encoding.decode(encoding.encode(text_item)[:MAX_TOKENS_PER_TEXT])
                        else:
                            # Fallback to simple word-based truncation
                            words = text_item.split()
                            truncated_text = ' '.join(words[:int(MAX_TOKENS_PER_TEXT/1.3)])  # Rough estimate
                        texts_to_process[texts_to_process.index(text_item)] = truncated_text

                all_embeddings = []
                total_tokens = 0

                # Process in batches
                for i in range(0, len(texts_to_process), batch_size):
                    batch = texts_to_process[i:i + batch_size]
                    response = await client.embeddings.create(
                        input=batch,
                        model=model,
                        encoding_format="float",
                        dimensions=3072)

                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    total_tokens += response.usage.total_tokens

                return EmbeddingResponse(
                    embedding=all_embeddings[0] if isinstance(text, str) else all_embeddings,
                    model=model,
                    usage={"total_tokens": total_tokens}
                )

            except Exception as e:
                logger.error(f"Error getting embeddings from {provider}: {str(e)}")
                if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self._clients:
                    logger.warning("Falling back to OpenAI embeddings")
                    return await self.embedding_request(text, provider=ModelProvider.OPENAI, batch_size=batch_size)
                raise

    def _get_default_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the default provider for a given operation.

        Args:
            operation: The model operation type

        Returns:
            ModelProvider enum representing the provider to use

        Raises:
            ValueError: If the specified operation is unknown
        """
        # Try to get provider based on operation
        if operation == ModelOperation.EMBEDDING:
            provider = ModelProvider.from_str(Config.get_default_embedding_provider())
        elif operation == ModelOperation.CHUNK_GENERATION:
            provider = ModelProvider.from_str(Config.get_default_chunk_provider())
        elif operation == ModelOperation.SUMMARIZATION:
            provider = ModelProvider.from_str(Config.get_default_summary_provider())
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Validate provider is configured
        if provider.value not in self._clients:
            available_providers = list(self._clients.keys())
            if not available_providers:
                # Instead of raising an error, log a warning and return OPENAI as a fallback
                # This will be handled later in the embedding_request method with mock embeddings
                logger.warning("No API providers are configured. Returning fallback provider for mock embeddings.")
                return ModelProvider.OPENAI  # Return a default provider that will trigger mock embeddings
            provider = ModelProvider.from_str(available_providers[0])
            logger.warning(f"Default provider not configured, using {provider.value}")

        return provider