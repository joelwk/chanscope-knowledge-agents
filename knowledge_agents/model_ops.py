import os
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union
from pydantic import BaseModel, Field
import openai
from openai import OpenAI
import logging
import yaml
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    OPENAI = "openai"
    GROK = "grok"
    VENICE = "venice"

    @classmethod
    def from_str(cls, value: str) -> "ModelProvider":
        try:
            return cls(value.lower())
        except ValueError:
            logger.warning(f"Invalid provider {value}, defaulting to OPENAI")
            return cls.OPENAI

class ModelOperation(str, Enum):
    EMBEDDING = "embedding"
    CHUNK_GENERATION = "chunk_generation"
    SUMMARIZATION = "summarization"

class ModelConfig(BaseModel):
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-large"
    openai_completion_model: str = "gpt-4o"

    # Grok Configuration
    grok_api_key: Optional[str] = None
    grok_embedding_model: Optional[str] = "grok-v1-embedding"
    grok_completion_model: Optional[str] = "grok-2-1212"

    # Venice Configuration
    venice_api_key: Optional[str] = None
    venice_summary_model: Optional[str] = "llama-3.1-405b"
    venice_chunk_model: Optional[str] = "dolphin-2.9.2-qwen2-72b"

    # Default providers for different operations
    default_embedding_provider: ModelProvider = ModelProvider.OPENAI
    default_chunk_provider: ModelProvider = ModelProvider.OPENAI
    default_summary_provider: ModelProvider = ModelProvider.OPENAI

    class Config:
        use_enum_values = True

class AppConfig(BaseModel):
    max_tokens: int = 8192
    chunk_size: int = 1000
    cache_enabled: bool = True
    batch_size: int = 100
    root_path: str
    all_data: str
    all_data_stratified_path: str
    knowledge_base: str
    sample_size: int = Field(gt=0)
    filter_date: Optional[str]

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

def load_config() -> Tuple[ModelConfig, AppConfig]:
    """Load configuration from environment variables."""
    model_config = ModelConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        openai_completion_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        grok_api_key=os.getenv("GROK_API_KEY"),
        grok_embedding_model=os.getenv("GROK_EMBEDDING_MODEL", "grok-v1-embedding"),
        grok_completion_model=os.getenv("GROK_MODEL", "grok-2-1212"),
        venice_api_key=os.getenv("VENICE_API_KEY"),
        venice_summary_model=os.getenv("VENICE_MODEL", "llama-3.1-405b"),
        venice_chunk_model=os.getenv("VENICE_CHUNK_MODEL", "dolphin-2.9.2-qwen2-72b"),
        default_embedding_provider=ModelProvider.from_str(os.getenv("DEFAULT_EMBEDDING_PROVIDER", "openai"))
    )

    app_config = AppConfig(
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        root_path=os.getenv("ROOT_PATH"),
        all_data=os.getenv("ALL_DATA"),
        all_data_stratified_path=os.getenv("ALL_DATA_STRATIFIED_PATH"),
        knowledge_base=os.getenv("KNOWLEDGE_BASE"),
        sample_size=int(os.getenv("SAMPLE_SIZE", "1000")),
        filter_date=os.getenv("FILTER_DATE"))
    return model_config, app_config

class KnowledgeAgent:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.models = self._initialize_clients()
        self.prompts = load_prompts()
        
    def _initialize_clients(self) -> Dict[str, OpenAI]:
        """Initialize model clients dynamically based on provided configuration."""
        clients = {}
        
        # Validate OpenAI configuration
        if not self.model_config.openai_api_key or not self.model_config.openai_api_key.strip():
            logging.warning("OpenAI API key is missing or empty")
        else:
            try:
                openai.api_key = self.model_config.openai_api_key
                clients[ModelProvider.OPENAI] = OpenAI(
                    api_key=self.model_config.openai_api_key,
                    max_retries=5,)
                logging.info("OpenAI client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Validate Grok configuration
        if self.model_config.grok_api_key and self.model_config.grok_api_key.strip():
            try:
                clients[ModelProvider.GROK] = OpenAI(
                    api_key=self.model_config.grok_api_key,
                    base_url="https://api.x.ai/v1",
                )
                logging.info("Grok client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Grok client: {str(e)}")
        # Validate Venice configuration
        if self.model_config.venice_api_key and self.model_config.venice_api_key.strip():
            try:
                clients[ModelProvider.VENICE] = OpenAI(
                    api_key=self.model_config.venice_api_key,
                    base_url="https://api.venice.ai/api/v1",)
                logging.info("Venice client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Venice client: {str(e)}")
        if not clients:
            raise ValueError("No API providers configured. Please check your API keys in config.ini")
        return clients

    def _get_client(self, provider: ModelProvider) -> OpenAI:
        """Retrieve the appropriate client for a provider."""
        client = self.models.get(provider)
        if not client:
            available_providers = list(self.models.keys())
            if not available_providers:
                raise ValueError("No API providers are configured")
            fallback_provider = available_providers[0]
            logging.warning(f"Provider {provider} not configured, falling back to {fallback_provider}")
            return self.models[fallback_provider]
        return client
    
    def _get_model_name(self, provider: ModelProvider, operation: ModelOperation) -> str:
        """Get the appropriate model name for a provider and operation type."""
        if operation == ModelOperation.EMBEDDING:
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_embedding_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_embedding_model or "grok-v1-embedding"
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        elif operation == ModelOperation.CHUNK_GENERATION:
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_completion_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_completion_model
            elif provider == ModelProvider.VENICE:
                return self.model_config.venice_chunk_model
            else:
                raise ValueError(f"Unsupported completion provider: {provider}")
        elif operation == ModelOperation.SUMMARIZATION:
            if provider == ModelProvider.OPENAI:
                return self.model_config.openai_completion_model
            elif provider == ModelProvider.GROK:
                return self.model_config.grok_completion_model
            elif provider == ModelProvider.VENICE:
                return self.model_config.venice_summary_model
            else:
                raise ValueError(f"Unsupported completion provider: {provider}")
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _get_default_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the default provider for a specific operation."""
        provider = None
        if operation == ModelOperation.EMBEDDING:
            provider = self.model_config.default_embedding_provider
            # Only OpenAI and Grok support embeddings
            if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
                provider = ModelProvider.OPENAI
        elif operation == ModelOperation.CHUNK_GENERATION:
            provider = self.model_config.default_chunk_provider
        elif operation == ModelOperation.SUMMARIZATION:
            provider = self.model_config.default_summary_provider
        else:
            raise ValueError(f"Unknown operation: {operation}")
        # Validate provider is configured
        if provider not in self.models:
            available_providers = list(self.models.keys())
            if not available_providers:
                raise ValueError("No API providers are configured. Please check your API keys in config.ini")
            provider = available_providers[0]
            logging.warning(f"Default provider not configured, using {provider}")
            
        return provider
    
    def generate_summary(
        self, 
        query: str, 
        results: str, 
        context: Optional[str] = None,
        temporal_context: Optional[Dict[str, str]] = None,
        provider: Optional[ModelProvider] = None
    ) -> str:
        """Generate a summary using the specified provider.
        
        Args:
            query: The original search query
            results: The combined chunk analysis results
            context: Additional analysis context
            temporal_context: Dictionary with start_date and end_date
            provider: The model provider to use
        """
        if provider is None:
            provider = self._get_default_provider(ModelOperation.SUMMARIZATION)
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.SUMMARIZATION)
        
        # Ensure temporal context is properly formatted
        if temporal_context is None:
            temporal_context = {
                "start_date": "Unknown",
                "end_date": "Unknown"
            }
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompts["system_prompts"]["objective_analysis"]["content"]
                    },
                    {
                        "role": "user",
                        "content": self.prompts["user_prompts"]["summary_generation"]["content"].format(
                            query=query,
                            temporal_context=f"Time Range: {temporal_context['start_date']} to {temporal_context['end_date']}",
                            context=context or "No additional context provided.",
                            results=results,
                            start_date=temporal_context['start_date'],
                            end_date=temporal_context['end_date']
                        )
                    }
                ],
                temperature=0.3,
                presence_penalty=0.2,
                frequency_penalty=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating summary with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning(f"Falling back to OpenAI for summary generation")
                return self.generate_summary(
                    query, 
                    results, 
                    context=context,
                    temporal_context=temporal_context,
                    provider=ModelProvider.OPENAI
                )
            raise
    
    def generate_chunks(
        self, 
        content: str,
        provider: Optional[ModelProvider] = None
    ) -> Dict[str, str]:
        """Generate chunks using the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.CHUNK_GENERATION)
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.CHUNK_GENERATION)
        
        try:
            response = client.chat.completions.create(
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
            sections = result.split("<generated_context>")
            if len(sections) > 1:
                analysis = sections[0].strip()
                context = sections[1].strip()
            else:
                analysis = result
                context = "No specific context generated."
                
            return {
                "analysis": analysis,
                "context": context
            }
            
        except Exception as e:
            logging.error(f"Error generating chunks with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning(f"Falling back to OpenAI for chunk generation")
                return self.generate_chunks(content, provider=ModelProvider.OPENAI)
            raise
        
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(2))
    def embedding_request(
        self,
        text: Union[str, List[str]],
        provider: Optional[ModelProvider] = None
    ) -> EmbeddingResponse:
        """Request embeddings from the specified provider."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.EMBEDDING)
            
        if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
            raise ValueError(f"Unsupported embedding provider: {provider}")
            
        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.EMBEDDING)
        
        try:
            response = client.embeddings.create(
                input=text,
                model=model,
                encoding_format="float",
                dimensions=3072)
            embeddings = [data.embedding for data in response.data]
            return EmbeddingResponse(
                embedding=embeddings[0] if isinstance(text, str) else embeddings,
                model=model,
                usage=response.usage.model_dump()
            )
            
        except Exception as e:
            logging.error(f"Error getting embeddings from {provider}: {str(e)}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning("Falling back to OpenAI embeddings")
                return self.embedding_request(text, provider=ModelProvider.OPENAI)
            raise