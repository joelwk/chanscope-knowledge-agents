"""Knowledge Agents Library

This library provides functionality for processing and analyzing text using various AI models.
It is designed to be configuration-agnostic and receive its settings from the application layer.
"""

import logging
from .model_ops import ModelProvider, ModelOperation, ModelConfig

logger = logging.getLogger(__name__)

# Export public interface
from .run import run_knowledge_agents
__all__ = ['ModelConfig', 'ModelProvider', 'ModelOperation', 'run_knowledge_agents']