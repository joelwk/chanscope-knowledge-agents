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
from config.base import BaseConfig
from knowledge_agents.utils import get_venice_character_slug
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
    def from_str(cls, value: str, **kwargs) -> 'ModelProvider':
        """Convert string to ModelProvider enum, handling carriage returns.
        
        Args:
            value: String value to convert to ModelProvider
            **kwargs: Additional keyword arguments (ignored)
        
        Returns:
            Corresponding ModelProvider enum member
            
        Raises:
            ValueError: If value is None or not a valid provider
        """
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
        logger.debug(f"Initializing ModelConfig with settings: {settings}")
        self._settings = settings
        
        # Initialize with default values for batch sizes
        self.embedding_batch_size = 10
        self.chunk_batch_size = 5
        self.summary_batch_size = 3
        
        # Initialize path settings with defaults
        self.root_data_path = 'data'
        self.stratified_path = 'data/stratified'
        self.temp_path = 'temp_files'
        
        # Get settings from base settings if not provided
        if not settings:
            settings = get_base_settings()

        # Get model settings
        model_settings = settings.get('model', {})

        # Model provider settings
        self.default_embedding_provider = model_settings.get('default_embedding_provider')
        self.default_chunk_provider = model_settings.get('default_chunk_provider')
        self.default_summary_provider = model_settings.get('default_summary_provider')
        self.venice_character_slug = model_settings.get('venice_character_slug')

        # Get path settings
        path_settings = settings.get('paths', {})

        # Path settings - convert to strings
        self.root_data_path = str(path_settings.get('root_data_path', self.root_data_path))
        self.stratified_path = str(path_settings.get('stratified', self.stratified_path))
        self.temp_path = str(path_settings.get('temp', self.temp_path))

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
        
        # Load batch sizes from centralized settings
        base_settings = BaseConfig.get_base_settings()['model']
        embedding_batch_size = base_settings.get('embedding_batch_size', 25)
        chunk_batch_size = base_settings.get('chunk_batch_size', 25)
        summary_batch_size = base_settings.get('summary_batch_size', 25)

        # Sanity checks for batch sizes
        if embedding_batch_size > 1000:
            logger.warning(f"Embedding batch size {embedding_batch_size} is unusually high, defaulting to 25")
            embedding_batch_size = 25
        if chunk_batch_size > 1000:
            logger.warning(f"Chunk batch size {chunk_batch_size} is unusually high, defaulting to 25")
            chunk_batch_size = 25
        if summary_batch_size > 1000:
            logger.warning(f"Summary batch size {summary_batch_size} is unusually high, defaulting to 25")
            summary_batch_size = 25

        self.embedding_batch_size = embedding_batch_size
        self.chunk_batch_size = chunk_batch_size
        self.summary_batch_size = summary_batch_size
        
        # Log configuration after everything is initialized
        logger.debug(f"ModelConfig initialized with paths: {self.paths}")
        logger.debug(f"Batch sizes: embedding={self.embedding_batch_size}, chunk={self.chunk_batch_size}, summary={self.summary_batch_size}")

    @property
    def paths(self) -> Dict[str, str]:
        """Get all configured filesystem paths."""
        # Return only the known filesystem paths, not any setting with 'path' in the name
        filesystem_paths = {
            'root_data_path': self.root_data_path,
            'stratified': self.stratified_path,
            'temp': self.temp_path
        }
        logger.debug(f"Retrieved filesystem paths: {filesystem_paths}")
        return filesystem_paths

    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create configuration from environment variables."""
        logger.debug("Loading ModelConfig from environment")
        try:
            base_settings = BaseConfig.get_base_settings()['model']
            settings = {
                'embedding_batch_size': base_settings.get('embedding_batch_size'),
                'chunk_batch_size': base_settings.get('chunk_batch_size'),
                'summary_batch_size': base_settings.get('summary_batch_size'),
                # ... other settings ...
            }
            logger.info(f"Loaded batch sizes from centralized settings: {settings}")
            return cls(**settings)
        except Exception as e:
            logger.error(f"Error loading environment settings: {e}")
            # Return a basic config with defaults
            logger.warning("Falling back to default configuration")
            return cls()

    @classmethod
    def from_request(cls, request):
        """Create configuration from a QueryRequest object.
        
        Properly handles provider parameters by converting them to strings rather than
        attempting to create provider objects directly, which would cause the
        "embedding_provider" keyword argument error.
        
        Args:
            request: A QueryRequest object containing configuration options
            
        Returns:
            A ModelConfig object configured with the request parameters
        """
        logger.debug("Building ModelConfig from request parameters")
        # Start with base settings
        settings = {
            'model': {},
            'processing': {},
            'paths': {}
        }
        
        # Add provider settings as strings
        if getattr(request, 'embedding_provider', None):
            settings['model']['default_embedding_provider'] = request.embedding_provider
        if getattr(request, 'chunk_provider', None):
            settings['model']['default_chunk_provider'] = request.chunk_provider
        if getattr(request, 'summary_provider', None):
            settings['model']['default_summary_provider'] = request.summary_provider
        if getattr(request, 'character_slug', None):
            settings['model']['venice_character_slug'] = request.character_slug
            
        # Add batch size settings
        if getattr(request, 'embedding_batch_size', None):
            settings['model']['embedding_batch_size'] = request.embedding_batch_size
        if getattr(request, 'chunk_batch_size', None):
            settings['model']['chunk_batch_size'] = request.chunk_batch_size
        if getattr(request, 'summary_batch_size', None):
            settings['model']['summary_batch_size'] = request.summary_batch_size
            
        # Add processing settings
        if getattr(request, 'filter_date', None):
            settings['processing']['filter_date'] = request.filter_date
        if getattr(request, 'sample_size', None):
            settings['processing']['sample_size'] = request.sample_size
        if getattr(request, 'max_workers', None):
            settings['processing']['max_workers'] = request.max_workers
            
        # Create and return ModelConfig
        return cls(**settings)

    def get_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the configured provider for a given operation."""
        provider_map = {
            ModelOperation.EMBEDDING: self.default_embedding_provider,
            ModelOperation.CHUNK_GENERATION: self.default_chunk_provider,
            ModelOperation.SUMMARIZATION: self.default_summary_provider
        }
        
        # If the provider is None, use a default (OPENAI)
        provider_value = provider_map[operation]
        if provider_value is None:
            return ModelProvider.OPENAI
            
        return ModelProvider(provider_value)

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
    """Load model configuration from environment variables."""
    try:
        # Load from environment
        logger.info("Loading model configuration from environment variables")
        env_config = ModelConfig.from_env()
        
        # Log batch size settings for debugging
        logger.info("Model configuration loaded:")
        logger.info(f"  Batch sizes: embedding={env_config.embedding_batch_size}, " +
                   f"chunk={env_config.chunk_batch_size}, summary={env_config.summary_batch_size}")
        logger.info(f"  Data paths: stratified={env_config.stratified_path}, " +
                   f"root={env_config.root_data_path}, temp={env_config.temp_path}")
        
        # Only create known filesystem directories, not API paths or other settings
        # that might contain 'path' in their name
        filesystem_paths = {
            'root_data_path': env_config.root_data_path,
            'stratified': env_config.stratified_path
            # temp_path is created on demand
        }
        
        # Ensure required directories exist
        for path_name, path in filesystem_paths.items():
            if path and not path.startswith('/api/'):  # Skip API paths
                path_obj = Path(path)
                if not path_obj.exists():
                    logger.info(f"Creating required directory: {path}")
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                    except PermissionError as pe:
                        logger.warning(f"Permission denied creating directory {path}: {pe}")
                        # Continue without failing - the app might still work if the directory already exists
                    except Exception as e:
                        logger.warning(f"Could not create directory {path}: {e}")
        
        return env_config
    except AttributeError as ae:
        logger.error(f"Configuration attribute error: {ae}")
        # Create a basic config with defaults as fallback
        logger.warning("Creating fallback configuration with default values")
        return ModelConfig()
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}")
        raise ModelConfigurationError(f"Failed to load model configuration: {str(e)}")

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
                    # Add a cache for models that don't support temperature
                    self._models_without_temperature = set()
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
            # Use Config from the top-level import
            base_settings = get_base_settings()['model']

            if provider == ModelProvider.OPENAI:
                return AsyncOpenAI(api_key=Config.get_openai_api_key())
            elif provider == ModelProvider.GROK:
                grok_key = Config.get_grok_api_key()
                if not grok_key:
                    raise ModelProviderError("Grok API key not found")
                return AsyncOpenAI(
                    api_key=grok_key,
                    base_url=base_settings.get('grok_api_base')
                )
            elif provider == ModelProvider.VENICE:
                venice_key = Config.get_venice_api_key()
                if not venice_key:
                    raise ModelProviderError("Venice API key not found")
                return AsyncOpenAI(
                    api_key=venice_key,
                    base_url=base_settings.get('venice_api_base'),
                )
            else:
                raise ModelProviderError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Error creating client for provider {provider}: {str(e)}")
            logger.error(f"Provider settings: {get_base_settings()['model']}")
            raise ModelProviderError(f"Failed to create client for {provider}: {str(e)}")

    def _validate_config(self, config: Optional[ModelConfig] = None) -> ModelConfig:
        """Validate and return configuration.
        
        Args:
            config: Optional configuration to validate
            
        Returns:
            Validated ModelConfig instance
        """
        if config is None:
            if self._config is None:
                self._config = load_config()
                logger.debug("Loaded new configuration")
            return self._config
        
        # If config is provided, validate it has required attributes
        required_attrs = ['embedding_batch_size', 'chunk_batch_size', 'summary_batch_size']
        for attr in required_attrs:
            if not hasattr(config, attr):
                logger.warning(f"Config missing required attribute: {attr}, using default")
                # Set default values if missing
                if attr == 'embedding_batch_size':
                    setattr(config, attr, 25)
                elif attr == 'chunk_batch_size':
                    setattr(config, attr, 5)
                elif attr == 'summary_batch_size':
                    setattr(config, attr, 3)
        
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

    # Add helper method to safely get environment variables
    @staticmethod
    def _get_env_model(provider_name, operation_name=None):
        """Safe fallback to get model name directly from environment variables."""
        if provider_name.lower() == 'openai':
            if operation_name and operation_name.lower() == 'chunk':
                return os.environ.get('OPENAI_CHUNK_MODEL') or os.environ.get('OPENAI_MODEL', 'gpt-4o')
            elif operation_name and operation_name.lower() == 'embedding':
                return os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
            else:
                return os.environ.get('OPENAI_MODEL', 'gpt-4o')
        elif provider_name.lower() == 'grok':
            if operation_name and operation_name.lower() == 'chunk':
                return os.environ.get('GROK_CHUNK_MODEL') or os.environ.get('GROK_MODEL', 'grok-2-1212')
            else:
                return os.environ.get('GROK_MODEL', 'grok-2-1212')
        elif provider_name.lower() == 'venice':
            if operation_name and operation_name.lower() == 'chunk':
                chunk_model = os.environ.get('VENICE_CHUNK_MODEL')
                if chunk_model:
                    return chunk_model
            return os.environ.get('VENICE_MODEL', 'deepseek-r1-671b')
        return None

    def _get_model_name(self, provider: ModelProvider, operation: ModelOperation) -> str:
        """Get model name based on provider and operation."""
        try:
            # Use Config directly without re-importing it
            if provider == ModelProvider.OPENAI:
                if operation == ModelOperation.EMBEDDING:
                    return Config.get_openai_embedding_model()
                elif operation == ModelOperation.CHUNK_GENERATION:
                    # Fallback to general model if chunk model getter fails
                    try:
                        return Config.get_openai_chunk_model()
                    except AttributeError as e:
                        logger.error(f"Error getting openai_chunk_model: {e}, falling back to general model")
                        return Config.get_openai_model()
                return Config.get_openai_model()
            elif provider == ModelProvider.GROK:
                if operation == ModelOperation.CHUNK_GENERATION:
                    # Fallback to general model if chunk model getter fails
                    try:
                        return Config.get_grok_chunk_model()
                    except AttributeError as e:
                        logger.error(f"Error getting grok_chunk_model: {e}, falling back to general model")
                        return Config.get_grok_model()
                return Config.get_grok_model()
            elif provider == ModelProvider.VENICE:
                if operation == ModelOperation.CHUNK_GENERATION:
                    # Check for venice chunk model
                    try:
                        chunk_model = Config.get_venice_chunk_model()
                        if chunk_model:
                            return chunk_model
                    except AttributeError as e:
                        logger.error(f"Error getting venice_chunk_model: {e}, falling back to general model")
                return Config.get_venice_model()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Error in _get_model_name for {provider} and operation {operation}: {e}")
            
            # Try direct environment variable access as last resort
            if operation == ModelOperation.CHUNK_GENERATION:
                env_model = self._get_env_model(provider.value, 'chunk')
            elif operation == ModelOperation.EMBEDDING:
                env_model = self._get_env_model(provider.value, 'embedding')
            else:
                env_model = self._get_env_model(provider.value)
                
            if env_model:
                logger.info(f"Using model {env_model} from environment variables for {provider.value}")
                return env_model
                
            # Final fallback to hardcoded values
            if provider == ModelProvider.OPENAI:
                return 'gpt-4o'
            elif provider == ModelProvider.GROK:
                return 'grok-2-1212'
            elif provider == ModelProvider.VENICE:
                return 'deepseek-r1-671b'
            raise ValueError(f"No model available for provider {provider}")

    def _prepare_model_params(self, provider: ModelProvider, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model parameters according to provider requirements.
        
        Different model providers support different parameters. This method ensures
        we only include supported parameters for each provider.
        
        Args:
            provider: The model provider (OpenAI, Grok, Venice)
            base_params: Base parameters including model and messages
            
        Returns:
            Dict containing appropriate parameters for the specified provider
        """
        model = base_params.get("model", "")
        request_params = base_params.copy()
        
        # Skip adding additional parameters if this model doesn't support them
        if model in self._models_without_temperature:
            return request_params
            
        # Provider-specific parameter configurations
        if provider == ModelProvider.GROK:
            # Grok only supports temperature, not presence/frequency penalty
            request_params.update({
                "temperature": request_params.get("temperature", 0.3)
            })
        elif provider == ModelProvider.OPENAI:
            # OpenAI supports all parameters
            request_params.update({
                "temperature": request_params.get("temperature", 0.3),
                "presence_penalty": request_params.get("presence_penalty", 0.2),
                "frequency_penalty": request_params.get("frequency_penalty", 0.2)
            })
        elif provider == ModelProvider.VENICE:
            # Venice parameters
            request_params.update({
                "temperature": request_params.get("temperature", 0.3),
                "presence_penalty": request_params.get("presence_penalty", 0.2),
                "frequency_penalty": request_params.get("frequency_penalty", 0.2)
            })
            
        return request_params
        
    async def _safe_model_call(self, client, request_params: Dict[str, Any]) -> Any:
        """Make a safe API call handling parameter-related errors gracefully.
        
        Tries to make the API call with all parameters first, then falls back to
        simpler parameter sets if needed based on error responses.
        
        Args:
            client: The API client to use
            request_params: Full set of request parameters
            
        Returns:
            The API response
            
        Raises:
            Exception: If the error is not related to parameters
        """
        try:
            return await client.chat.completions.create(**request_params)
        except Exception as e:
            error_str = str(e)
            model = request_params.get("model", "unknown")
            
            # Handle temperature-related errors
            if "temperature" in error_str and ("unsupported" in error_str or "not supported" in error_str):
                logger.warning(f"Temperature not supported by model {model}, retrying without temperature parameters")
                # Remember this model doesn't support temperature for future requests
                self._models_without_temperature.add(model)
                # Remove temperature-related parameters
                for param in ["temperature", "presence_penalty", "frequency_penalty"]:
                    if param in request_params:
                        del request_params[param]
                # Retry without temperature parameters
                return await client.chat.completions.create(**request_params)
            # Handle other parameter-related errors
            elif any(x in error_str.lower() for x in ["not supported", "invalid argument", "presencepenalty", "frequencypenalty"]):
                logger.warning(f"Parameter not supported by model {model}: {error_str}, retrying with basic parameters")
                # Strip out all optional parameters
                basic_params = {
                    "model": model,
                    "messages": request_params["messages"]
                }
                # Retry with only basic parameters
                return await client.chat.completions.create(**basic_params)
            else:
                # If it's not a parameter-related error, re-raise
                raise

    async def generate_summary(
        self, 
        query: str, 
        results: str, 
        context: Optional[str] = None,
        temporal_context: Optional[Dict[str, str]] = None,
        provider: Optional[ModelProvider] = None,
        character_slug: Optional[str] = None
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

            # Prepare API request params
            request_params = {
                "model": model,
                "messages": messages
            }
            
            # Add temperature and penalties based on provider
            request_params = self._prepare_model_params(provider, request_params)
            
            # Add venice_parameters if using Venice provider and character_slug is specified
            # pisagor-ai https://venice.ai/c/pisagor-ai?ref=KWgrlE
            # coinrotatorai-1 https://venice.ai/c/coinrotatorai-1?ref=rvp5n5
            if provider == ModelProvider.VENICE:
                # Use provided character_slug or get default from config
                char_slug = character_slug
                if char_slug is None:
                    char_slug = get_venice_character_slug()
                
                if char_slug:
                    # Pass character_slug in the messages parameter structure
                    for message in request_params["messages"]:
                        if "parameters" not in message:
                            message["parameters"] = {}
                        message["parameters"]["character_slug"] = char_slug

            # Make the API call with safe error handling
            response = await self._safe_model_call(client, request_params)
            
            # Process the response
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary with {provider.value}: {str(e)}")
            
            # If the first provider fails, try fallback to OpenAI
            if provider != ModelProvider.OPENAI:
                logger.warning(f"Falling back to OpenAI for summary generation")
                return await self.generate_summary(
                    query=query,
                    results=results,
                    context=context,
                    temporal_context=temporal_context,
                    provider=ModelProvider.OPENAI
                )
            
            # If we're already using OpenAI or all fallbacks fail, raise the error
            raise SummarizationError(f"Failed to generate summary: {str(e)}") from e

    async def generate_chunks(
        self, 
        content: str,
        provider: Optional[ModelProvider] = None,
        character_slug: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate chunks using the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.CHUNK_GENERATION)

        client = await self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.CHUNK_GENERATION)

        async with self._chunk_lock:
            try:
                # Prepare API request params
                request_params = {
                    "model": model,
                    "messages": [
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
                    ]
                }
                
                # Prepare parameters based on provider
                request_params = self._prepare_model_params(provider, request_params)
                
                # Add venice_parameters if using Venice provider and character_slug is specified
                if provider == ModelProvider.VENICE:
                    # Use provided character_slug or get default from config
                    char_slug = character_slug
                    if char_slug is None:
                        char_slug = get_venice_character_slug()
                    
                    if char_slug:
                        # Pass character_slug in the messages parameter structure
                        for message in request_params["messages"]:
                            if "parameters" not in message:
                                message["parameters"] = {}
                            message["parameters"]["character_slug"] = char_slug

                # Make the API call with safe error handling
                response = await self._safe_model_call(client, request_params)
                
                # Process the response
                content = response.choices[0].message.content
                
                # Try to parse the response as JSON
                try:
                    result = json.loads(content)
                    
                    # Ensure all keys are present
                    if not all(key in result for key in ["thread_id", "chunks", "summary"]):
                        logger.warning(f"Unexpected structure in chunks response: {content[:50]}...")
                        # Format as minimum viable result
                        return {
                            "thread_id": self._generate_thread_id(content),
                            "chunks": [content],  # Use full content as single chunk
                            "summary": content[:100]  # Use first 100 chars as summary
                        }
                    
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse chunks response as JSON: {content[:50]}...")
                    # Return a fallback response
                    return {
                        "thread_id": self._generate_thread_id(content),
                        "chunks": [content],  # Use full content as single chunk
                        "summary": content[:100]  # Use first 100 chars as summary
                    }
            
            except Exception as e:
                logger.error(f"Error generating chunks with {provider.value}: {str(e)}")
                
                # If the first provider fails, try fallback to OpenAI
                if provider != ModelProvider.OPENAI:
                    logger.warning(f"Falling back to OpenAI for chunk generation")
                    return await self.generate_chunks(content, provider=ModelProvider.OPENAI)
                
                # If we're already using OpenAI or all fallbacks fail, raise the error
                raise ChunkGenerationError(f"Failed to generate chunks: {str(e)}") from e

    def _optimize_batch_size(self, contents: List[str], default_batch_size: int) -> int:
        """Dynamically optimize batch size based on content length.
        
        For very long content items, reduce batch size to avoid rate limits and token limits.
        
        Args:
            contents: List of content strings to process
            default_batch_size: Default batch size from configuration
            
        Returns:
            Optimized batch size
        """
        if not contents:
            return default_batch_size
            
        # Calculate average content length
        avg_length = sum(len(content) for content in contents) / len(contents)
        
        # Adjust batch size based on content length
        if avg_length > 10000:  # Very long content
            return max(1, min(3, default_batch_size))
        elif avg_length > 5000:  # Moderately long content
            return max(1, min(5, default_batch_size))
        elif avg_length > 2000:  # Average content
            return max(1, min(10, default_batch_size))
        else:  # Short content
            return default_batch_size
            
    async def generate_chunks_batch(
        self,
        contents: List[str],
        provider: Optional[ModelProvider] = None,
        chunk_batch_size: Optional[int] = None,
        character_slug: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Generate content chunks for a batch of text contents.
        
        Args:
            contents: List of text contents to process
            provider: Optional model provider override
            chunk_batch_size: Optional batch size override
            character_slug: Optional character slug for Venice provider
            
        Returns:
            List of chunked contents
        """
        if not contents:
            return []
            
        # Ensure we have a valid config
        self._config = self._validate_config(self._config)
        
        # Get provider (use default if not specified)
        provider = provider or self._get_default_provider(ModelOperation.CHUNK_GENERATION)
        
        # Get appropriate batch size from config if not specified
        config_batch_size = self._config.chunk_batch_size
        
        # Use provided batch size, or config batch size, or default
        batch_size = chunk_batch_size or config_batch_size or 5
        
        # Optimize batch size based on content length if needed
        optimized_batch_size = self._optimize_batch_size(contents, batch_size)
        
        # Log debug information about batching
        logger.info(f"Chunk generation batch settings - Requested: {chunk_batch_size}, " +
                   f"Config: {config_batch_size}, Optimized: {optimized_batch_size}, Using: {optimized_batch_size}")
        logger.info(f"Content statistics - Count: {len(contents)}, " + 
                   f"Avg Length: {sum(len(c) for c in contents) / len(contents):.1f} chars, " +
                   f"Max Length: {max(len(c) for c in contents)} chars")
        
        # Actual number of batches
        num_batches = (len(contents) + optimized_batch_size - 1) // optimized_batch_size
        
        logger.info(f"Starting generate_chunks_batch with {len(contents)} items, batch_size={optimized_batch_size}, resulting in {num_batches} batches")
        
        results = []
        total_batches = (len(contents) + optimized_batch_size - 1) // optimized_batch_size
        
        for i in range(0, len(contents), optimized_batch_size):
            batch = contents[i:i + optimized_batch_size]
            batch_num = i // optimized_batch_size + 1
            
            logger.info(f"Processing chunk batch {batch_num}/{total_batches}: {len(batch)} items")
            
            # Process batch with proper error handling
            try:
                batch_results = await asyncio.gather(
                    *[self.generate_chunks(content, provider=provider, character_slug=character_slug) for content in batch],
                    return_exceptions=True
                )
                
                # Handle any exceptions in the batch
                processed_results = []
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in chunk generation (item {i+j}): {str(result)}")
                        processed_results.append(None)  # Use None for failed items
                    else:
                        processed_results.append(result)
                        
                results.extend(processed_results)
                logger.info(f"Completed chunk batch {batch_num}/{total_batches}, total results: {len(results)}/{len(contents)}")
                
            except Exception as e:
                logger.error(f"Error processing chunk batch {batch_num}: {str(e)}")
                # Add None results for the entire batch on catastrophic failure
                results.extend([None] * len(batch))
        
        return results

    async def generate_summaries_batch(
        self,
        queries: List[str],
        results_list: List[str],
        contexts: Optional[List[str]] = None,
        temporal_contexts: Optional[List[Dict[str, str]]] = None,
        provider: Optional[ModelProvider] = None,
        summary_batch_size: Optional[int] = None,
        character_slug: Optional[str] = None
    ) -> List[str]:
        """Generate summaries for multiple queries in batches.
        
        Args:
            queries: List of query strings
            results_list: List of results strings (one per query)
            contexts: Optional list of context strings (one per query)
            temporal_contexts: Optional list of temporal context dicts (one per query)
            provider: Optional model provider override
            summary_batch_size: Optional batch size override
            character_slug: Optional character slug for Venice provider
            
        Returns:
            List of summary strings (one per query)
        """
        if not queries or not results_list:
            return []
            
        if len(queries) != len(results_list):
            raise ValueError(f"Number of queries ({len(queries)}) must match number of results ({len(results_list)})")
            
        if contexts and len(contexts) != len(queries):
            raise ValueError(f"Number of contexts ({len(contexts)}) must match number of queries ({len(queries)})")
            
        if temporal_contexts and len(temporal_contexts) != len(queries):
            raise ValueError(f"Number of temporal contexts ({len(temporal_contexts)}) must match number of queries ({len(queries)})")
        
        # Ensure we have a valid config
        self._config = self._validate_config(self._config)
        
        # Get provider (use default if not specified)
        provider = provider or self._get_default_provider(ModelOperation.SUMMARIZATION)
        
        # Get appropriate batch size from config if not specified
        config_batch_size = self._config.summary_batch_size
        
        # Use provided batch size, or config batch size, or default
        batch_size = summary_batch_size or config_batch_size or 3
        
        # Log batch information
        logger.info(f"Summary generation batch settings - Requested: {summary_batch_size}, " +
                   f"Config: {config_batch_size}, Using: {batch_size}")
        logger.info(f"Processing {len(queries)} summaries in batches of {batch_size}")
        
        # Prepare empty contexts if not provided
        if not contexts:
            contexts = [None] * len(queries)
            
        if not temporal_contexts:
            temporal_contexts = [None] * len(queries)
        
        # Process in batches
        results = []
        total_batches = (len(queries) + batch_size - 1) // batch_size
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_results = results_list[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            batch_temporal_contexts = temporal_contexts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing summary batch {batch_num}/{total_batches}: {len(batch_queries)} items")
            
            # Process batch with proper error handling
            try:
                batch_summaries = await asyncio.gather(
                    *[self.generate_summary(
                        query=query,
                        results=result,
                        context=context,
                        temporal_context=temp_context,
                        provider=provider,
                        character_slug=character_slug
                    ) for query, result, context, temp_context in zip(
                        batch_queries, batch_results, batch_contexts, batch_temporal_contexts
                    )],
                    return_exceptions=True
                )
                
                # Handle any exceptions in the batch
                processed_summaries = []
                for j, summary in enumerate(batch_summaries):
                    if isinstance(summary, Exception):
                        logger.error(f"Error in summary generation (item {i+j}): {str(summary)}")
                        processed_summaries.append(f"Error generating summary: {str(summary)}")
                    else:
                        processed_summaries.append(summary)
                        
                results.extend(processed_summaries)
                logger.info(f"Completed summary batch {batch_num}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error processing summary batch {batch_num}: {str(e)}")
                # Add error messages for the entire batch on catastrophic failure
                results.extend([f"Error generating summary: {str(e)}"] * len(batch_queries))
        
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
            # Check if mock embeddings are explicitly requested via Config
            use_mock_embeddings = Config.use_mock_embeddings()
            mock_provider = Config.get_default_embedding_provider().lower() == 'mock'
            
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

            # If mock embeddings are explicitly requested, generate them immediately
            if use_mock_embeddings or mock_provider:
                logger.info(f"Generating mock embeddings for {len(texts_to_process)} texts (USE_MOCK_EMBEDDINGS={use_mock_embeddings}, mock_provider={mock_provider})")
                mock_embeddings = generate_mock_embeddings(texts_to_process)
                return EmbeddingResponse(
                    embedding=mock_embeddings if len(mock_embeddings) > 1 else mock_embeddings[0],
                    model="mock-embedding-model",
                    usage={"prompt_tokens": 0, "total_tokens": 0}
                )

            try:
                client = await self._get_client(provider)
                model = self._get_model_name(provider, ModelOperation.EMBEDDING)

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
        # Check if mock embeddings are explicitly requested via Config
        use_mock_embeddings = Config.use_mock_embeddings()
        if use_mock_embeddings and operation == ModelOperation.EMBEDDING:
            logger.info("Using mock embeddings as requested by USE_MOCK_EMBEDDINGS environment variable")
            # Return a provider that will trigger mock embeddings
            return ModelProvider.OPENAI  # This will be handled in embedding_request with mock embeddings
            
        # Try to get provider based on operation
        if operation == ModelOperation.EMBEDDING:
            provider_str = Config.get_default_embedding_provider()
            # Check if provider is explicitly set to 'mock'
            if provider_str.lower() == 'mock':
                logger.info("Mock embedding provider explicitly configured")
                return ModelProvider.OPENAI  # This will be handled in embedding_request with mock embeddings
            provider = ModelProvider.from_str(provider_str)
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

    def _generate_thread_id(self, content: str) -> str:
        """Generate a thread ID for content that doesn't have one.
        
        Args:
            content: The text content to generate an ID for
            
        Returns:
            A unique thread ID based on content hash
        """
        # Create a hash of the content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Create a thread ID with a timestamp prefix for uniqueness
        import time
        timestamp = int(time.time())
        return f"gen_{timestamp}_{content_hash}"