"""Error classes for the API."""
from typing import Any, Dict, Optional
from fastapi import status

class APIError(Exception):
    """Base exception for API errors with structured logging."""
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

class ProcessingError(APIError):
    """Exception for processing errors."""
    def __init__(
        self, 
        message: str,
        operation: Optional[str] = None,
        resource: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {
            "operation": operation,
            "resource": resource
        } if operation or resource else {}
        super().__init__(
            message=message,
            status_code=500,
            error_code="PROCESSING_ERROR",
            details=details,
            original_error=original_error
        ) 