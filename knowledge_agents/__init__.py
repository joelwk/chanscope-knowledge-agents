"""Knowledge Agents Library

This library provides functionality for processing and analyzing text using various AI models.
It is designed to be configuration-agnostic and receive its settings from the application layer.
"""

from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto

class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    GROK = "grok"
    VENICE = "venice"

class ModelOperation(str, Enum):
    """Types of model operations"""
    EMBEDDING = "embedding"
    CHUNK_GENERATION = "chunk_generation"
    SUMMARIZATION = "summarization"

@dataclass
class KnowledgeAgentConfig:
    """Configuration for Knowledge Agents library.
    
    This class encapsulates all configuration needed by the library,
    allowing it to be instantiated by the application layer without
    directly accessing environment variables.
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
        # Convert string paths to Path objects
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
    def from_env(cls, env_vars: Dict[str, str]) -> 'KnowledgeAgentConfig':
        """Create configuration from environment variables.
        
        This factory method allows the application layer to pass environment
        variables, but the library itself doesn't load them directly.
        """
        return cls(
            root_path=env_vars.get('ROOT_PATH', './data'),
            knowledge_base_path=env_vars.get('KNOWLEDGE_BASE', './data/knowledge_base.csv'),
            all_data_path=env_vars.get('ALL_DATA', './data/all_data.csv'),
            stratified_data_path=env_vars.get('ALL_DATA_STRATIFIED_PATH', './data/stratified'),
            temp_path=env_vars.get('PATH_TEMP', './temp_files'),
            batch_size=int(env_vars.get('BATCH_SIZE', 100)),
            max_workers=int(env_vars.get('MAX_WORKERS', 4)) if env_vars.get('MAX_WORKERS') else None,
            sample_size=int(env_vars.get('SAMPLE_SIZE', 2500)),
            providers={
                ModelOperation.EMBEDDING: ModelProvider(env_vars.get('DEFAULT_EMBEDDING_PROVIDER', 'openai')),
                ModelOperation.CHUNK_GENERATION: ModelProvider(env_vars.get('DEFAULT_CHUNK_PROVIDER', 'openai')),
                ModelOperation.SUMMARIZATION: ModelProvider(env_vars.get('DEFAULT_SUMMARY_PROVIDER', 'openai'))
            }
        )

# Export public interface
from .run import run_knowledge_agents
__all__ = ['KnowledgeAgentConfig', 'ModelProvider', 'ModelOperation', 'run_knowledge_agents']