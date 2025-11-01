"""API routes module - DEPRECATED.

This module is kept for backward compatibility during migration.
All endpoints have been migrated to domain-specific routers in api/routers/.
This file will be removed in a future version.

Migration status:
- All health endpoints -> api/routers/health.py
- All data endpoints -> api/routers/data.py  
- All query endpoints -> api/routers/query.py
- All embedding endpoints -> api/routers/embeddings.py
- All admin endpoints -> api/routers/admin.py
- Shared utilities -> api/routers/shared.py
"""

from fastapi import APIRouter
from config.logging_config import get_logger

logger = get_logger(__name__)

# Empty router kept for backward compatibility
# All endpoints have been migrated to domain-specific routers
router = APIRouter()

# Note: This router is registered in app.py but contains no endpoints.
# It will be removed once all references are updated.
