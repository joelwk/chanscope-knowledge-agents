import os
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union
from pydantic import BaseModel
from openai import AsyncOpenAI
import logging
import yaml
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pathlib import Path
from config.settings import Config
import json
import tiktoken

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

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
    max_tokens: int 
    chunk_size: int
    cache_enabled: bool
    sample_size: int
    root_data_path: str
    stratified_path: str
    knowledge_base: str
    filter_date: Optional[str]

class EmbeddingResponse(BaseModel):
    """Standardized embedding response across providers."""
    embedding: Union[List[float], List[List[float]]]
    model: str
    usage: Dict[str, int]

class ModelConfig:
    """Configuration class for model operations."""
    def __init__(
        self,
        path_settings: Dict[str, str],
        model_settings: Dict[str, Any],
        api_settings: Dict[str, Any]
    ):
        """Initialize model configuration."""
        # Path settings
        self.root_data_path = path_settings['root_data_path']
        self.stratified = path_settings['stratified']
        self.knowledge_base = path_settings['knowledge_base']
        self.temp = path_settings['temp']
        
        # Model settings
        self.embedding_model = model_settings['embedding_model']
        self.chunk_model = model_settings['chunk_model']
        self.summary_model = model_settings['summary_model']
        
        # API settings
        self.openai_api_key = api_settings.get('openai_api_key')
        self.grok_api_key = api_settings.get('grok_api_key')
        self.venice_api_key = api_settings.get('venice_api_key')

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
        path_settings=Config.get_paths(),
        model_settings=Config.get_model_settings(),
        api_settings=Config.get_api_settings()
    )

    # Get settings from Config
    processing_settings = Config.get_processing_settings()
    path_settings = Config.get_paths()
    sample_settings = Config.get_sample_settings()

    app_config = AppConfig(
        max_tokens=processing_settings['max_tokens'],
        chunk_size=processing_settings['chunk_size'],
        cache_enabled=processing_settings['cache_enabled'],
        sample_size=sample_settings['default_sample_size'],
        root_data_path=path_settings['root_data_path'],
        stratified_path=path_settings['stratified'],
        knowledge_base=path_settings['knowledge_base'],
        filter_date=processing_settings['filter_date']
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

        # Get settings
        sample_settings = Config.get_sample_settings()
        processing_settings = Config.get_processing_settings()

        # Set default values
        self.sample_size = sample_settings['default_sample_size']
        self.max_workers = processing_settings['max_workers']
        self.cache_enabled = processing_settings['cache_enabled']

    def _initialize_clients(self):
        """Initialize API clients based on available credentials."""
        clients = {}

        # Check for OpenAI configuration
        if self.model_config.openai_api_key:
            try:
                clients['openai'] = AsyncOpenAI(api_key=self.model_config.openai_api_key)
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
        model_settings = Config.get_model_settings()
        
        if operation == ModelOperation.EMBEDDING:
            if provider == ModelProvider.OPENAI:
                return model_settings['embedding_model']
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        elif operation == ModelOperation.CHUNK_GENERATION:
            if provider == ModelProvider.OPENAI:
                return model_settings['chunk_model']
            elif provider == ModelProvider.GROK:
                return model_settings['grok_model']
            elif provider == ModelProvider.VENICE:
                return model_settings['venice_chunk_model']
            else:
                raise ValueError(f"Unsupported chunk generation provider: {provider}")
        elif operation == ModelOperation.SUMMARIZATION:
            if provider == ModelProvider.OPENAI:
                return model_settings['chunk_model']  # Use same model for summarization
            elif provider == ModelProvider.GROK:
                return model_settings['grok_model']
            elif provider == ModelProvider.VENICE:
                return model_settings['venice_model']
            else:
                raise ValueError(f"Unsupported summarization provider: {provider}")
        else:
            raise ValueError(f"Unknown operation: {operation}")

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

        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.SUMMARIZATION)

        # Ensure temporal context is properly formatted
        if temporal_context is None:
            temporal_context = {
                "start_date": "Unknown",
                "end_date": "Unknown"
            }

        try:
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
            chunk_results = json.loads(results)
            if not isinstance(chunk_results, list):
                chunk_results = [chunk_results]

            # Convert chunks to text format for token counting
            chunks_text = json.dumps(chunk_results, indent=2)
            chunks_tokens = len(encoding.encode(chunks_text))

            if chunks_tokens > available_tokens:
                # Use create_chunks to manage token size
                from .inference_ops import create_chunks
                chunks = create_chunks(chunks_text, available_tokens, encoding)

                # Take the first chunk that fits
                if chunks:
                    chunks_text = chunks[0]["text"]
                else:
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
        except Exception as e:
            logging.error(f"Error generating summary with {provider}: {e}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning(f"Falling back to OpenAI for summary generation")
                return await self.generate_summary(
                    query, 
                    results, 
                    context=context,
                    temporal_context=temporal_context,
                    provider=ModelProvider.OPENAI
                )
            raise

    async def generate_chunks(
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
                return await self.generate_chunks(content, provider=ModelProvider.OPENAI)
            raise ValueError(f"Failed to generate chunks with {provider}: {str(e)}")

    async def generate_chunks_batch(
        self,
        contents: List[str],
        provider: Optional[ModelProvider] = None,
        chunk_batch_size: int = 20  # OpenAI recommends 20 requests per batch
    ) -> List[Dict[str, str]]:
        """Generate chunks for multiple contents in batches."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.CHUNK_GENERATION)

        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.CHUNK_GENERATION)
        results = []

        # Process in batches
        for i in range(0, len(contents), chunk_batch_size):
            batch = contents[i:i + chunk_batch_size]
            try:
                # Create messages for each content in batch
                messages = [
                    {
                        "role": "system",
                        "content": self.prompts["system_prompts"]["generate_chunks"]["content"]
                    }
                ]
                for content in batch:
                    messages.append({
                        "role": "user",
                        "content": self.prompts["user_prompts"]["text_chunk_summary"]["content"].format(
                            content=content
                        )
                    })

                # Make batch request
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )

                # Process responses
                for choice in response.choices:
                    result = self._process_chunk_response(choice.message.content)
                    results.append(result)

            except Exception as e:
                logger.error(f"Error in batch chunk generation: {e}")
                # Fall back to individual processing if batch fails
                for content in batch:
                    try:
                        result = await self.generate_chunks(content, provider)
                        results.append(result)
                    except Exception as inner_e:
                        logger.error(f"Error in fallback chunk generation: {inner_e}")
                        results.append(None)

        return results

    async def generate_summaries_batch(
        self,
        queries: List[str],
        results_list: List[str],
        contexts: Optional[List[str]] = None,
        temporal_contexts: Optional[List[Dict[str, str]]] = None,
        provider: Optional[ModelProvider] = None,
        summary_batch_size: int = 20  # OpenAI recommends 20 requests per batch
    ) -> List[str]:
        """Generate summaries for multiple queries in batches."""
        if provider is None:
            provider = self._get_default_provider(ModelOperation.SUMMARIZATION)

        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.SUMMARIZATION)

        # Initialize optional parameters
        if contexts is None:
            contexts = [None] * len(queries)
        if temporal_contexts is None:
            temporal_contexts = [{"start_date": "Unknown", "end_date": "Unknown"}] * len(queries)

        summaries = []

        # Process in batches
        for i in range(0, len(queries), summary_batch_size):
            batch_queries = queries[i:i + summary_batch_size]
            batch_results = results_list[i:i + summary_batch_size]
            batch_contexts = contexts[i:i + summary_batch_size]
            batch_temporal = temporal_contexts[i:i + summary_batch_size]

            try:
                # Create messages for each query in batch
                messages = []
                for q, r, c, t in zip(batch_queries, batch_results, batch_contexts, batch_temporal):
                    messages.extend([
                        {
                            "role": "system",
                            "content": self.prompts["system_prompts"]["objective_analysis"]["content"]
                        },
                        {
                            "role": "user",
                            "content": self.prompts["user_prompts"]["summary_generation"]["content"].format(
                                query=q,
                                temporal_context=f"Time Range: {t['start_date']} to {t['end_date']}",
                                context=c or "No additional context provided.",
                                results=r,
                                start_date=t['start_date'],
                                end_date=t['end_date']
                            )
                        }
                    ])

                # Make batch request
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    presence_penalty=0.2,
                    frequency_penalty=0.2
                )

                # Process responses
                batch_summaries = [choice.message.content for choice in response.choices]
                summaries.extend(batch_summaries)

            except Exception as e:
                logger.error(f"Error in batch summary generation: {e}")
                # Fall back to individual processing if batch fails
                for q, r, c, t in zip(batch_queries, batch_results, batch_contexts, batch_temporal):
                    try:
                        summary = await self.generate_summary(q, r, c, t, provider)
                        summaries.append(summary)
                    except Exception as inner_e:
                        logger.error(f"Error in fallback summary generation: {inner_e}")
                        summaries.append(None)

        return summaries

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
        """Get embeddings for text using specified provider.

        This implementation follows OpenAI's best practices for batching:
        - Respects the passed batch_size parameter for number of items per batch
        - Counts tokens to ensure we don't exceed OpenAI's limits (max 8191 tokens per text)
        - Falls back to token-based batching if no batch_size specified
        - Properly handles both single and batch requests
        """
        if provider is None:
            provider = self._get_default_provider(ModelOperation.EMBEDDING)

        if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        client = self._get_client(provider)
        model = self._get_model_name(provider, ModelOperation.EMBEDDING)

        try:
            # Ensure text is properly formatted
            if isinstance(text, str):
                input_text = text.strip()
                if not input_text:
                    raise ValueError("Empty text input")
                texts_to_process = [input_text]
            elif isinstance(text, list):
                # Filter and clean list input
                texts_to_process = [str(t).strip() for t in text if t and str(t).strip()]
                if not texts_to_process:
                    raise ValueError("No valid text inputs in list")
            else:
                raise ValueError(f"Invalid input type: {type(text)}")

            # Get token counts for validation
            encoding = tiktoken.get_encoding("cl100k_base")
            token_counts = [len(encoding.encode(t)) for t in texts_to_process]

            # Validate token counts (OpenAI limit is 8191 per text)
            MAX_TOKENS_PER_TEXT = 8191
            for text_item, token_count in zip(texts_to_process, token_counts):
                if token_count > MAX_TOKENS_PER_TEXT:
                    logger.warning(f"Text exceeds token limit ({token_count} > {MAX_TOKENS_PER_TEXT}). Truncating...")
                    # Truncate text to fit within limits
                    truncated_text = encoding.decode(encoding.encode(text_item)[:MAX_TOKENS_PER_TEXT])
                    texts_to_process[texts_to_process.index(text_item)] = truncated_text

            all_embeddings = []

            if batch_size:
                # Use the specified batch_size for batching
                for i in range(0, len(texts_to_process), batch_size):
                    batch = texts_to_process[i:i + batch_size]
                    response = await client.embeddings.create(
                        input=batch,
                        model=model,
                        encoding_format="float",
                        dimensions=3072)
                    all_embeddings.extend([data.embedding for data in response.data])
            else:
                # Fall back to token-based batching if no batch_size specified
                tokens_per_batch = 2048
                current_batch = []
                current_tokens = 0

                for text_item, token_count in zip(texts_to_process, token_counts):
                    if current_tokens + token_count > tokens_per_batch and current_batch:
                        response = await client.embeddings.create(
                            input=current_batch,
                            model=model,
                            encoding_format="float",
                            dimensions=3072)
                        all_embeddings.extend([data.embedding for data in response.data])
                        current_batch = []
                        current_tokens = 0

                    current_batch.append(text_item)
                    current_tokens += token_count

                # Process any remaining items
                if current_batch:
                    response = await client.embeddings.create(
                        input=current_batch,
                        model=model,
                        encoding_format="float",
                        dimensions=3072)
                    all_embeddings.extend([data.embedding for data in response.data])

            # Return appropriate format based on input type
            return EmbeddingResponse(
                embedding=all_embeddings[0] if isinstance(text, str) else all_embeddings,
                model=model,
                usage=response.usage.model_dump()  # Using last response's usage
            )

        except Exception as e:
            logging.error(f"Error getting embeddings from {provider}: {str(e)}")
            if provider != ModelProvider.OPENAI and ModelProvider.OPENAI in self.models:
                logging.warning("Falling back to OpenAI embeddings")
                return await self.embedding_request(text, provider=ModelProvider.OPENAI)
            raise

    def _get_client(self, provider: ModelProvider) -> AsyncOpenAI:
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
        model_settings = Config.get_model_settings()
        
        if operation == ModelOperation.EMBEDDING:
            provider = ModelProvider.from_str(model_settings['default_embedding_provider'])
            # Only OpenAI and Grok support embeddings
            if provider not in [ModelProvider.OPENAI, ModelProvider.GROK]:
                provider = ModelProvider.OPENAI
        elif operation == ModelOperation.CHUNK_GENERATION:
            provider = ModelProvider.from_str(model_settings['default_chunk_provider'])
        elif operation == ModelOperation.SUMMARIZATION:
            provider = ModelProvider.from_str(model_settings['default_summary_provider'])
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