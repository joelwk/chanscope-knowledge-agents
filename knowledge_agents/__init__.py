"""Knowledge Agents Library

This library provides functionality for processing and analyzing text using various AI models.
It is designed to be configuration-agnostic and receive its settings from the application layer.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Remove direct imports from model_ops to prevent circular imports
# These classes will be re-exported at the bottom of this file

logger = logging.getLogger(__name__)


class KnowledgeDocument:
    """
    Represents a document or piece of content with metadata, designed for knowledge processing.
    
    Attributes:
        thread_id (str): The unique identifier for the thread/document
        posted_date_time (str): The datetime when the document was posted
        text_clean (str): The cleaned text content of the document
        embedding (Optional[list]): Optional embedding vector for the document
    """
    
    def __init__(
        self, 
        thread_id: str, 
        posted_date_time: str, 
        text_clean: str,
        embedding: Optional[list] = None
    ):
        self.thread_id = thread_id
        self.posted_date_time = posted_date_time
        self.text_clean = text_clean
        self.embedding = embedding
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeDocument':
        """
        Create a KnowledgeDocument instance from a dictionary.
        
        Args:
            data: Dictionary containing document data with keys:
                 thread_id, posted_date_time, text_clean, and optionally embedding
        
        Returns:
            KnowledgeDocument: A new KnowledgeDocument instance
        """
        return cls(
            thread_id=str(data.get('thread_id', '')),
            posted_date_time=str(data.get('posted_date_time', '')),
            text_clean=str(data.get('text_clean', '')),
            embedding=data.get('embedding')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the KnowledgeDocument to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing the document data
        """
        return {
            'thread_id': self.thread_id,
            'posted_date_time': self.posted_date_time,
            'text_clean': self.text_clean,
            'embedding': self.embedding
        }
    
    def __str__(self) -> str:
        """String representation of the KnowledgeDocument."""
        preview = self.text_clean[:50] + "..." if len(self.text_clean) > 50 else self.text_clean
        return f"KnowledgeDocument(thread_id={self.thread_id}, posted_date_time={self.posted_date_time}, text_preview='{preview}')"
    
    def __repr__(self) -> str:
        """Detailed representation of the KnowledgeDocument."""
        return self.__str__()
    
    @property
    def posted_datetime(self) -> Optional[datetime]:
        """
        Parse the posted_date_time string into a datetime object.
        
        Returns:
            Optional[datetime]: Datetime object or None if parsing fails
        """
        try:
            return datetime.fromisoformat(self.posted_date_time.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            try:
                # Try alternative formats
                from dateutil import parser
                return parser.parse(self.posted_date_time)
            except:
                return None

# Export public interface - use lazy imports to avoid circular dependencies
# This approach defers the imports until they're actually needed
from .run import run_inference

# Create placeholder names that will be imported only when used
__all__ = ['ModelConfig', 'ModelProvider', 'ModelOperation', 'run_inference', 'KnowledgeDocument']

# Define __getattr__ to lazily load modules only when accessed
def __getattr__(name):
    if name in ('ModelConfig', 'ModelProvider', 'ModelOperation'):
        from .model_ops import ModelConfig, ModelProvider, ModelOperation
        if name == 'ModelConfig': 
            return ModelConfig
        elif name == 'ModelProvider':
            return ModelProvider
        elif name == 'ModelOperation':
            return ModelOperation
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")