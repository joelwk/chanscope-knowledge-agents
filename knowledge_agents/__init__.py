"""Knowledge Agents Library

This library provides functionality for processing and analyzing text using various AI models.
It is designed to be configuration-agnostic and receive its settings from the application layer.
"""

from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
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
        root_path: str = Config.ROOT_PATH,
        all_data_path: str = Config.ALL_DATA,
        stratified_data_path: str = Config.ALL_DATA_STRATIFIED_PATH,
        knowledge_base_path: str = Config.KNOWLEDGE_BASE,
        sample_size: int = Config.SAMPLE_SIZE,  # Number of items to process
        embedding_batch_size: int = Config.EMBEDDING_BATCH_SIZE,  # Token-based batching (2048)
        chunk_batch_size: int = Config.CHUNK_BATCH_SIZE,  # OpenAI recommended (20)
        summary_batch_size: int = Config.SUMMARY_BATCH_SIZE,  # OpenAI recommended (20)
        max_workers: Optional[int] = Config.MAX_WORKERS,
        providers: Optional[Dict[ModelOperation, ModelProvider]] = None
    ):
        """Initialize knowledge agent configuration.
        
        Args:
            root_path: Root path for data storage
            all_data_path: Path to all data
            stratified_data_path: Path to stratified data
            knowledge_base_path: Path to knowledge base
            sample_size: Number of items to process in each batch (sample-based batching)
            embedding_batch_size: Maximum tokens per batch for embedding operations (default: 2048)
            chunk_batch_size: Maximum items per batch for chunk generation (default: 20)
            summary_batch_size: Maximum items per batch for summary generation (default: 20)
            max_workers: Maximum number of worker threads
            providers: Dictionary mapping operations to model providers
        """
        self.root_path = root_path
        self.all_data_path = all_data_path
        self.stratified_data_path = stratified_data_path
        self.knowledge_base_path = knowledge_base_path
        self.sample_size = sample_size
        self.embedding_batch_size = embedding_batch_size
        self.chunk_batch_size = chunk_batch_size
        self.summary_batch_size = summary_batch_size
        self.max_workers = max_workers
        self.providers = providers or {}

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate paths
        for path_attr in ['root_path', 'all_data_path', 'stratified_data_path', 'knowledge_base_path']:
            path = getattr(self, path_attr)
            if not isinstance(path, (str, Path)):
                raise ValueError(f"{path_attr} must be a string or Path")

        # Validate numeric values
        for num_attr in ['sample_size', 'embedding_batch_size', 'chunk_batch_size', 'summary_batch_size']:
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
        return cls(
            root_path=Config.ROOT_PATH,
            all_data_path=Config.ALL_DATA,
            stratified_data_path=Config.ALL_DATA_STRATIFIED_PATH,
            knowledge_base_path=Config.KNOWLEDGE_BASE,
            sample_size=Config.SAMPLE_SIZE,
            embedding_batch_size=Config.EMBEDDING_BATCH_SIZE,
            chunk_batch_size=Config.CHUNK_BATCH_SIZE,
            summary_batch_size=Config.SUMMARY_BATCH_SIZE,
            max_workers=Config.MAX_WORKERS,
            providers={
                ModelOperation.EMBEDDING: ModelProvider(Config.EMBEDDING_PROVIDER),
                ModelOperation.CHUNK_GENERATION: ModelProvider(Config.CHUNK_PROVIDER),
                ModelOperation.SUMMARIZATION: ModelProvider(Config.SUMMARY_PROVIDER)
            }
        )

    @classmethod
    def from_settings(cls, settings: Any) -> 'KnowledgeAgentConfig':
        """Create configuration from settings object."""
        return cls(
            root_path=getattr(settings, 'ROOT_PATH', Config.ROOT_PATH),
            all_data_path=getattr(settings, 'ALL_DATA', Config.ALL_DATA),
            stratified_data_path=getattr(settings, 'ALL_DATA_STRATIFIED_PATH', Config.ALL_DATA_STRATIFIED_PATH),
            knowledge_base_path=getattr(settings, 'KNOWLEDGE_BASE', Config.KNOWLEDGE_BASE),
            sample_size=int(getattr(settings, 'SAMPLE_SIZE', Config.SAMPLE_SIZE)),
            embedding_batch_size=int(getattr(settings, 'EMBEDDING_BATCH_SIZE', Config.EMBEDDING_BATCH_SIZE)),
            chunk_batch_size=int(getattr(settings, 'CHUNK_BATCH_SIZE', Config.CHUNK_BATCH_SIZE)),
            summary_batch_size=int(getattr(settings, 'SUMMARY_BATCH_SIZE', Config.SUMMARY_BATCH_SIZE)),
            max_workers=int(getattr(settings, 'MAX_WORKERS', Config.MAX_WORKERS)),
            providers={
                ModelOperation.EMBEDDING: ModelProvider(getattr(settings, 'EMBEDDING_PROVIDER', Config.EMBEDDING_PROVIDER)),
                ModelOperation.CHUNK_GENERATION: ModelProvider(getattr(settings, 'CHUNK_PROVIDER', Config.CHUNK_PROVIDER)),
                ModelOperation.SUMMARIZATION: ModelProvider(getattr(settings, 'SUMMARY_PROVIDER', Config.SUMMARY_PROVIDER))
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