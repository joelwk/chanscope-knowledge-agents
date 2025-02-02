"""Knowledge Agents Library

This library provides functionality for processing and analyzing text using various AI models.
It is designed to be configuration-agnostic and receive its settings from the application layer.
"""

from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from config.settings import Config

from .model_ops import ModelProvider, ModelOperation

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeAgentConfig:
    """Configuration for knowledge agent operations.

    This class handles two types of batching:
    1. Sample-based batching: Controls the number of items processed (sample_size)
    2. Token-based batching: Controls the batch sizes for API calls based on token limits
        - embedding_batch_size: Max 2048 tokens per batch for embeddings
        - chunk_batch_size: Max 20 items per batch for chunk generation (OpenAI recommended)
        - summary_batch_size: Max 20 items per batch for summary generation (OpenAI recommended)
    """
    def __init__(
        self,
        root_data_path: str = None,
        stratified_data_path: str = None,
        knowledge_base_path: str = None,
        sample_size: int = None,
        embedding_batch_size: int = None,
        chunk_batch_size: int = None,
        summary_batch_size: int = None,
        max_workers: Optional[int] = None,
        providers: Optional[Dict[ModelOperation, ModelProvider]] = None
    ):
        """Initialize knowledge agent configuration."""
        # Get centralized settings
        paths = Config.get_paths()
        model_settings = Config.get_model_settings()
        sample_settings = Config.get_sample_settings()
        processing_settings = Config.get_processing_settings()

        # Initialize paths
        self.root_data_path = root_data_path or paths['root_data_path']
        self.stratified_data_path = stratified_data_path or paths['stratified_path']
        self.knowledge_base_path = knowledge_base_path or paths['knowledge_base']

        # Initialize batch sizes
        self.sample_size = sample_size or sample_settings['default_sample_size']
        self.embedding_batch_size = embedding_batch_size or model_settings['embedding_batch_size']
        self.chunk_batch_size = chunk_batch_size or model_settings['chunk_batch_size']
        self.summary_batch_size = summary_batch_size or model_settings['summary_batch_size']
        self.max_workers = max_workers or processing_settings['max_workers']

        # Initialize providers
        self.providers = providers or {
            ModelOperation.EMBEDDING: ModelProvider(model_settings['default_embedding_provider']),
            ModelOperation.CHUNK_GENERATION: ModelProvider(model_settings['default_chunk_provider']),
            ModelOperation.SUMMARIZATION: ModelProvider(model_settings['default_summary_provider'])
        }

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Get settings for validation
        sample_settings = Config.get_sample_settings()

        # Validate sample size
        if self.sample_size > sample_settings['max_sample_size']:
            logger.warning(f"Sample size {self.sample_size} exceeds maximum of {sample_settings['max_sample_size']}. Setting to maximum.")
            self.sample_size = sample_settings['max_sample_size']
        elif self.sample_size < sample_settings['min_sample_size']:
            logger.warning(f"Sample size {self.sample_size} below minimum of {sample_settings['min_sample_size']}. Setting to minimum.")
            self.sample_size = sample_settings['min_sample_size']

        # Validate paths
        for path_attr in ['root_data_path','stratified_data_path', 'knowledge_base_path']:
            path = getattr(self, path_attr)
            if not isinstance(path, (str, Path)):
                raise ValueError(f"{path_attr} must be a string or Path")

        # Validate numeric values
        for num_attr in ['embedding_batch_size', 'chunk_batch_size', 'summary_batch_size']:
            value = getattr(self, num_attr)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{num_attr} must be a positive integer")

        # Validate max_workers
        if self.max_workers is not None and (not isinstance(self.max_workers, int) or self.max_workers <= 0):
            raise ValueError("max_workers must be None or a positive integer")

        # Validate providers
        if not isinstance(self.providers, dict):
            raise ValueError("providers must be a dictionary")
        for op, provider in self.providers.items():
            if not isinstance(op, ModelOperation):
                raise ValueError("Provider keys must be ModelOperation instances")
            if not isinstance(provider, ModelProvider):
                raise ValueError("Provider values must be ModelProvider instances")

    @classmethod
    def from_env(cls) -> 'KnowledgeAgentConfig':
        """Create configuration from environment variables through Config."""
        paths = Config.get_paths()
        model_settings = Config.get_model_settings()
        sample_settings = Config.get_sample_settings()
        processing_settings = Config.get_processing_settings()
        
        return cls(
            root_data_path=paths['root_data_path'],
            stratified_data_path=paths['stratified'],
            knowledge_base_path=paths['knowledge_base'],
            sample_size=sample_settings['default_sample_size'],
            embedding_batch_size=model_settings['embedding_batch_size'],
            chunk_batch_size=model_settings['chunk_batch_size'],
            summary_batch_size=model_settings['summary_batch_size'],
            max_workers=processing_settings['max_workers'],
            providers={
                ModelOperation.EMBEDDING: ModelProvider(model_settings['default_embedding_provider']),
                ModelOperation.CHUNK_GENERATION: ModelProvider(model_settings['default_chunk_provider']),
                ModelOperation.SUMMARIZATION: ModelProvider(model_settings['default_summary_provider'])
            }
        )

    @classmethod
    def from_settings(cls, settings: Any) -> 'KnowledgeAgentConfig':
        """Create configuration from settings object."""
        paths = Config.get_paths()
        model_settings = Config.get_model_settings()
        sample_settings = Config.get_sample_settings()
        processing_settings = Config.get_processing_settings()
        
        return cls(
            root_data_path=getattr(settings, 'root_data_path', paths['root_data_path']),
            stratified_data_path=getattr(settings, 'stratified_path', paths['stratified']),
            knowledge_base_path=getattr(settings, 'knowledge_base_path', paths['knowledge_base']),
            sample_size=getattr(settings, 'sample_size', sample_settings['default_sample_size']),

            embedding_batch_size=getattr(settings, 'embedding_batch_size', model_settings['embedding_batch_size']),
            chunk_batch_size=getattr(settings, 'chunk_batch_size', model_settings['chunk_batch_size']),
            summary_batch_size=getattr(settings, 'summary_batch_size', model_settings['summary_batch_size']),
            max_workers=getattr(settings, 'max_workers', processing_settings['max_workers']),
            providers={
                ModelOperation.EMBEDDING: ModelProvider(model_settings['default_embedding_provider']),
                ModelOperation.CHUNK_GENERATION: ModelProvider(model_settings['default_chunk_provider']),
                ModelOperation.SUMMARIZATION: ModelProvider(model_settings['default_summary_provider'])
            }
        )

    def get_batch_config(self) -> Dict[str, int]:
        """Get the current batch configuration.

        Returns:
            Dict containing all batch-related settings
        """
        return {
            "sample_size": self.sample_size,
            "embedding_batch_size": self.embedding_batch_size,
            "chunk_batch_size": self.chunk_batch_size,
            "summary_batch_size": self.summary_batch_size
        }

# Export public interface
from .run import run_knowledge_agents
__all__ = ['KnowledgeAgentConfig', 'ModelProvider', 'ModelOperation', 'run_knowledge_agents']