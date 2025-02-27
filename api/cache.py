"""Simple in-memory caching system for API responses and embeddings.

NOTE: This is a self-contained implementation without external dependencies.
Future enhancements could include:
1. Redis integration for distributed caching
2. Prometheus metrics for monitoring
3. More sophisticated eviction policies

For now, this provides a reliable in-memory cache with basic metrics.
"""
import time
import json
import logging
import asyncio
import functools
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

# Simple metrics counters
class SimpleCounter:
    """Simple counter for tracking metrics without external dependencies."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.value = 0
    
    def inc(self, amount: float = 1):
        """Increment the counter by the specified amount."""
        self.value += amount
    
    def get_value(self) -> float:
        """Get the current counter value."""
        return self.value

# Define metrics counters
CACHE_HITS = SimpleCounter('cache_hits_total', 'Number of cache hits')
CACHE_MISSES = SimpleCounter('cache_misses_total', 'Number of cache misses')
CACHE_ERRORS = SimpleCounter('cache_errors_total', 'Number of cache errors')

class InMemoryCache:
    """Thread-safe in-memory caching system."""
    
    def __init__(self, ttl: int = 3600):
        """Initialize the cache with TTL in seconds.
        
        Args:
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.ttl = ttl  # Default TTL: 1 hour
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_lock = asyncio.Lock()
                
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found
        """
        normalized_key = self._normalize_key(key)
        
        try:
            # Check memory cache
            async with self._memory_lock:
                if normalized_key in self._memory_cache:
                    entry = self._memory_cache[normalized_key]
                    # Check if entry is still valid
                    if entry.get('expires_at', 0) > time.time():
                        CACHE_HITS.inc()
                        logger.debug(f"Cache hit for {normalized_key}")
                        return entry.get('value')
                    else:
                        # Remove expired entry
                        del self._memory_cache[normalized_key]
            
            CACHE_MISSES.inc()
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            CACHE_ERRORS.inc()
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Optional custom TTL in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if value is None:
            return False
            
        normalized_key = self._normalize_key(key)
        expires_at = time.time() + (ttl or self.ttl)
        
        try:
            # Store in memory cache
            async with self._memory_lock:
                self._memory_cache[normalized_key] = {
                    'value': value,
                    'expires_at': expires_at
                }
                logger.debug(f"Stored in cache: {normalized_key}")
                
                # Cleanup policy: remove oldest entries if too many
                if len(self._memory_cache) > 1000:  # Maximum cache size
                    # Sort by expiration time and remove oldest 20%
                    sorted_keys = sorted(
                        self._memory_cache.keys(),
                        key=lambda k: self._memory_cache[k].get('expires_at', 0)
                    )
                    for old_key in sorted_keys[:int(len(sorted_keys) * 0.2)]:
                        del self._memory_cache[old_key]
                    
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            CACHE_ERRORS.inc()
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        normalized_key = self._normalize_key(key)
        
        try:
            deleted = False
            
            # Delete from memory cache
            async with self._memory_lock:
                if normalized_key in self._memory_cache:
                    del self._memory_cache[normalized_key]
                    deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            CACHE_ERRORS.inc()
            return False
    
    async def clear(self) -> bool:
        """Clear the entire cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear memory cache
            async with self._memory_lock:
                self._memory_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            CACHE_ERRORS.inc()
            return False
    
    def _normalize_key(self, key: str) -> str:
        """Normalize a cache key.
        
        Args:
            key: Raw cache key
            
        Returns:
            Normalized cache key
        """
        # If key is too long, hash it to get a fixed length key
        if len(key) > 200:
            return f"cache:{hashlib.md5(key.encode()).hexdigest()}"
        return f"cache:{key}"
    
    def _json_encoder(self, obj):
        """JSON encoder for special types.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Create singleton cache instance
cache = InMemoryCache(ttl=3600)  # Default 1 hour TTL

def async_cache(ttl: Optional[int] = None):
    """Decorator for caching async function results.
    
    Args:
        ttl: Optional custom TTL in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name, args, and kwargs
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached
                
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            return result
            
        return wrapper
    return decorator

# Specific embeddings cache functions
async def get_cached_embedding(text: str) -> Optional[List[float]]:
    """Get cached embedding for a text string.
    
    Args:
        text: Text to get embedding for
        
    Returns:
        Cached embedding or None
    """
    # Normalize text and create cache key
    normalized_text = text.strip().lower()
    cache_key = f"embedding:{hashlib.md5(normalized_text.encode()).hexdigest()}"
    
    return await cache.get(cache_key)

async def cache_embedding(text: str, embedding: Union[List[float], np.ndarray]) -> bool:
    """Cache embedding for a text string.
    
    Args:
        text: Text string
        embedding: Embedding vector
        
    Returns:
        True if cached successfully, False otherwise
    """
    # Normalize text and create cache key
    normalized_text = text.strip().lower() 
    cache_key = f"embedding:{hashlib.md5(normalized_text.encode()).hexdigest()}"
    
    # Convert numpy array to list for storage
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
        
    # Use 1 day TTL for embeddings
    return await cache.set(cache_key, embedding, 86400) 