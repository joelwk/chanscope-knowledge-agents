"""API routers module.

This module contains all API route handlers organized by domain.
"""

from .health import router as health_router
from .query import router as query_router
from .data import router as data_router
from .embeddings import router as embeddings_router
from .admin import router as admin_router

__all__ = [
    "health_router",
    "query_router",
    "data_router",
    "embeddings_router",
    "admin_router",
]

