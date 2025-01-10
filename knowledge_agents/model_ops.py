import os
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union
from pydantic import BaseModel, Field
from openai import OpenAI
import logging
import yaml
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pathlib import Path
from config.settings import Config

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    OPENAI = "openai"
    GROK = "grok"
    VENICE = "venice"
    
    @classmethod
    def from_str(cls, value: str) -> "ModelProvider":
        """Convert string to ModelProvider enum."""
        if isinstance(value, cls):
            return value
            
        try:
            return cls(value.lower() if isinstance(value, str) else value)
        except ValueError:
            raise ValueError(f"Unknown model provider: {value}")
class ModelOperation(str, Enum):
    EMBEDDING = "embedding"
    CHUNK_GENERATION = "chunk_generation"
    SUMMARIZATION = "summarization"
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
    
class ModelConfig:
    """Configuration for model operations."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_embedding_model: Optional[str] = None,
        openai_completion_model: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        grok_embedding_model: Optional[str] = None,
        grok_completion_model: Optional[str] = None,
        venice_api_key: Optional[str] = None,
        venice_summary_model: Optional[str] = None,
        venice_chunk_model: Optional[str] = None,
        default_embedding_provider: Optional[str] = None
    ):
        """Initialize with API keys and model configurations."""
        # OpenAI settings - ensure we have a valid API key
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required but not provided")
            
        self.openai_embedding_model = openai_embedding_model or Config.OPENAI_EMBEDDING_MODEL
        self.openai_completion_model = openai_completion_model or Config.OPENAI_MODEL
        
        # Grok settings
        self.grok_api_key = grok_api_key or Config.GROK_API_KEY
        self.grok_embedding_model = grok_embedding_model or Config.GROK_EMBEDDING_MODEL
        self.grok_completion_model = grok_completion_model or Config.GROK_MODEL
        
        # Venice settings
        self.venice_api_key = venice_api_key or Config.VENICE_API_KEY
        self.venice_summary_model = venice_summary_model or Config.VENICE_MODEL
        self.venice_chunk_model = venice_chunk_model or Config.VENICE_CHUNK_MODEL
        
        # Default provider
        self.default_embedding_provider = ModelProvider.from_str(
            default_embedding_provider or Config.DEFAULT_EMBEDDING_PROVIDER
        )
        
        # General settings
        self.max_tokens = Config.MAX_TOKENS
        self.chunk_size = Config.CHUNK_SIZE
        self.cache_enabled = Config.CACHE_ENABLED
        self.batch_size = Config.DEFAULT_BATCH_SIZE
        
        # Paths
        self.root_path = Config.ROOT_PATH
        self.all_data = Config.ALL_DATA
        self.all_data_stratified_path = Config.ALL_DATA_STRATIFIED_PATH
        self.knowledge_base = Config.KNOWLEDGE_BASE
        self.sample_size = Config.DEFAULT_BATCH_SIZE
        self.filter_date = Config.FILTER_DATE

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
    """Load configuration using Config class from settings."""
    # Create model config using Config class values
    model_config = ModelConfig(
        openai_api_key=Config.OPENAI_API_KEY,
        openai_embedding_model=Config.OPENAI_EMBEDDING_MODEL,
        openai_completion_model=Config.OPENAI_MODEL,
        grok_api_key=Config.GROK_API_KEY,
        grok_embedding_model=Config.GROK_EMBEDDING_MODEL,
        grok_completion_model=Config.GROK_MODEL,
        venice_api_key=Config.VENICE_API_KEY,
        venice_summary_model=Config.VENICE_MODEL,
        venice_chunk_model=Config.VENICE_CHUNK_MODEL,
        default_embedding_provider=Config.DEFAULT_EMBEDDING_PROVIDER
    )

    app_config = AppConfig(
        max_tokens=Config.MAX_TOKENS,
        chunk_size=Config.CHUNK_SIZE,
        cache_enabled=Config.CACHE_ENABLED,
        batch_size=Config.DEFAULT_BATCH_SIZE,
        root_path=Config.ROOT_PATH,
        all_data=Config.ALL_DATA,
        all_data_stratified_path=Config.ALL_DATA_STRATIFIED_PATH,
        knowledge_base=Config.KNOWLEDGE_BASE,
        sample_size=Config.DEFAULT_BATCH_SIZE,
        filter_date=Config.FILTER_DATE
    )
    return model_config, app_config

class KnowledgeAgent:
    """Agent for handling model operations and API interactions."""
    
    def __init__(self):
        """Initialize the KnowledgeAgent using settings from Config."""
        # Load configuration
        self.model_config, self.app_config = load_config()
        
        # Initialize API clients
        self.models = self._initialize_clients()
        
        # Load prompts
        self.prompts = load_prompts()
        
    def _initialize_clients(self):
        """Initialize API clients based on available credentials."""
        clients = {}
        
        # Check for OpenAI configuration
        if self.model_config.openai_api_key:
            try:
                clients['openai'] = OpenAI(api_key=self.model_config.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {str(e)}")

        # Check for Grok configuration
        if self.model_config.grok_api_key:
            try:
                clients['grok'] = {'api_key': self.model_config.grok_api_key}
                logger.info("Grok client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok client: {str(e)}")

        # Check for Venice configuration
        if self.model_config.venice_api_key:
            try:
                clients['venice'] = {'api_key': self.model_config.venice_api_key}
                logger.info("Venice client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Venice client: {str(e)}")

        if not clients:
            raise ValueError(
                "No API providers configured. Please set at least one of the following in settings:\n"
                "- OPENAI_API_KEY (Required)\n"
                "- GROK_API_KEY (Optional)\n"
                "- VENICE_API_KEY (Optional)")
        return clients
        
    def _get_model_name(self, provider: ModelProvider, operation: ModelOperation) -> str:
        """Get the appropriate model name for a provider and operation type."""
        if operation == ModelOperation.EMBEDDING:
            if provider == ModelProvider.OPENAI:
                return Config.OPENAI_EMBEDDING_MODEL
            elif provider == ModelProvider.GROK:
                return Config.GROK_EMBEDDING_MODEL
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        elif operation == ModelOperation.CHUNK_GENERATION:
            if provider == ModelProvider.OPENAI:
                return Config.OPENAI_MODEL
            elif provider == ModelProvider.GROK:
                return Config.GROK_MODEL
            elif provider == ModelProvider.VENICE:
                return Config.VENICE_CHUNK_MODEL
            else:
                raise ValueError(f"Unsupported chunk generation provider: {provider}")
        elif operation == ModelOperation.SUMMARIZATION:
            if provider == ModelProvider.OPENAI:
                return Config.OPENAI_MODEL
            elif provider == ModelProvider.GROK:
                return Config.GROK_MODEL
            elif provider == ModelProvider.VENICE:
                return Config.VENICE_MODEL
            else:
                raise ValueError(f"Unsupported summarization provider: {provider}")
        else:
            raise ValueError(f"Unknown operation: {operation}")

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
                            # Extract metrics from format: (t, metric, value, confidence)
                            metric_str = line[line.find("(")+1:line.find(")")].strip()
                            parts = [p.strip().strip("'\"") for p in metric_str.split(',')]                            
                            # Skip header lines
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
                        # Remove leading dash/asterisk and strip whitespace
                        content = line.lstrip('-* ').strip()
                        if content:  # Only add non-empty lines
                            context_elements[current_section].append(content)
                
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
                # If no signal context found, try to parse as basic analysis
                logger.warning("No signal_context found in response, using basic format")
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
            
        except Exception as e:
            logger.error(f"Error generating chunks with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logger.warning(f"Falling back to OpenAI for chunk generation")
                return self.generate_chunks(content, provider=ModelProvider.OPENAI)
            raise ValueError(f"Failed to generate chunks with {provider}: {str(e)}")
        
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

    def _get_client(self, provider: ModelProvider) -> OpenAI:
        """Retrieve the appropriate client for a provider."""
        if not isinstance(provider, ModelProvider):
            provider = ModelProvider.from_str(provider)
            
        client = self.models.get(provider.value)
        if not client:
            available_providers = list(self.models.keys())
            if not available_providers:
                raise ValueError("No API providers are configured")
            fallback_provider = available_providers[0]
            logging.warning(f"Provider {provider.value} not configured, falling back to {fallback_provider}")
            return self.models[fallback_provider]
        return client
    
    def _get_default_provider(self, operation: ModelOperation) -> ModelProvider:
        """Get the default provider for a specific operation."""
        provider = None
        if operation == ModelOperation.EMBEDDING:
            provider = ModelProvider.from_str(Config.DEFAULT_EMBEDDING_PROVIDER)
            # Only OpenAI and Grok support embeddings
            if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
                provider = ModelProvider.OPENAI
        elif operation == ModelOperation.CHUNK_GENERATION:
            provider = ModelProvider.from_str(Config.DEFAULT_CHUNK_PROVIDER)
        elif operation == ModelOperation.SUMMARIZATION:
            provider = ModelProvider.from_str(Config.DEFAULT_SUMMARY_PROVIDER)
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # Validate provider is configured
        if provider.value not in self.models:
            available_providers = list(self.models.keys())
            if not available_providers:
                raise ValueError("No API providers are configured. Please check your API keys in settings")
            provider = ModelProvider.from_str(available_providers[0])
            logging.warning(f"Default provider not configured, using {provider.value}")
            
        return provider