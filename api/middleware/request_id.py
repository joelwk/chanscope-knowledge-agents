"""Request ID middleware for tracing requests across the application."""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each request.
    
    This middleware:
    - Generates a unique UUID for each request
    - Adds X-Request-ID header to the response
    - Stores the request ID in request.state for logging
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate or use existing request ID from header
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Store in request state for logging access
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

