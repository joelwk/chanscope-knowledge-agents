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
    """Configuration for Knowledge Agents library.
    
    This class encapsulates all configuration needed by the library,
    providing a type-safe interface that is independent of the application's
    configuration system. It can be instantiated using either:
    - from_settings(): Creates instance from application's Config object
    - from_env(): Creates instance directly from environment variables
    """
    # Data paths
    root_path: Path
    knowledge_base_path: Path
    all_data_path: Path
    stratified_data_path: Path
    temp_path: Path
    
    # Processing settings
    batch_size: int = 100
    max_workers: Optional[int] = None
    sample_size: int = 2500
    
    # Model settings
    providers: Dict[ModelOperation, ModelProvider] = None
    model_settings: Dict[str, Any] = field(default_factory=lambda: {
        'max_tokens': 8192,
        'chunk_size': 1000,
        'cache_enabled': True
    })
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert string paths to Path objects if they aren't already
        self.root_path = Path(self.root_path)
        self.knowledge_base_path = Path(self.knowledge_base_path)
        self.all_data_path = Path(self.all_data_path)
        self.stratified_data_path = Path(self.stratified_data_path)
        self.temp_path = Path(self.temp_path)
        
        # Create directories if they don't exist
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.stratified_data_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize default providers if none provided
        if self.providers is None:
            self.providers = {
                ModelOperation.EMBEDDING: ModelProvider.OPENAI,
                ModelOperation.CHUNK_GENERATION: ModelProvider.GROK,
                ModelOperation.SUMMARIZATION: ModelProvider.VENICE
            }

    @classmethod
    def from_env(cls, env: Dict[str, str]) -> 'KnowledgeAgentConfig':
        """Create configuration from environment variables using Config class.
        
        Args:
            env: Dictionary of environment variables (typically os.environ)
            
        Returns:
            KnowledgeAgentConfig: A new instance with settings from environment.
        """
        # Use Config class which already handles all environment variables properly
        config = Config()
        return cls.from_settings(config)

    @classmethod
    def from_settings(cls, config) -> 'KnowledgeAgentConfig':
        """Create configuration from settings.Config instance.
        
        Args:
            config: An instance of config.settings.Config containing all application settings.
            
        Returns:
            KnowledgeAgentConfig: A new instance with settings from the config object.
        """
        paths = config.get_data_paths()
        processing = config.get_processing_settings()
        providers = config.get_provider_settings()
        
        return cls(
            root_path=paths['root'],
            knowledge_base_path=paths['knowledge_base'],
            all_data_path=paths['all_data'],
            stratified_data_path=paths['stratified'],
            temp_path=paths['temp'],
            batch_size=processing['batch_size'],
            max_workers=processing['max_workers'],
            sample_size=processing.get('sample_size', 2500),  # Use default if not in config
            providers={
                ModelOperation.EMBEDDING: ModelProvider(providers['embedding_provider']),
                ModelOperation.CHUNK_GENERATION: ModelProvider(providers['chunk_provider']),
                ModelOperation.SUMMARIZATION: ModelProvider(providers['summary_provider'])
            },
            model_settings={
                'max_tokens': processing['max_tokens'],
                'chunk_size': processing['chunk_size'],
                'cache_enabled': processing['cache_enabled']
            }
        )

# Export public interface
from .run import run_knowledge_agents
__all__ = ['KnowledgeAgentConfig', 'ModelProvider', 'ModelOperation', 'run_knowledge_agents']