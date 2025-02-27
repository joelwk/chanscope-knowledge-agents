from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standardized error response model."""
    detail: str
    error_code: str
    status_code: int
    additional_info: Optional[Dict[str, Any]] = None

class BaseAppException(HTTPException):
    """Base exception for all application exceptions."""
    def __init__(
        self,
        detail: str,
        error_code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.additional_info = additional_info
        super().__init__(status_code=status_code, detail=detail)

    def to_response(self) -> ErrorResponse:
        """Convert exception to standardized error response."""
        return ErrorResponse(
            detail=self.detail,
            error_code=self.error_code,
            status_code=self.status_code,
            additional_info=self.additional_info
        )

class ValidationError(BaseAppException):
    """Raised when input validation fails."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            additional_info=additional_info
        )

class AuthenticationError(BaseAppException):
    """Raised when authentication fails."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            additional_info=additional_info
        )

class AuthorizationError(BaseAppException):
    """Raised when authorization fails."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            additional_info=additional_info
        )

class ResourceNotFoundError(BaseAppException):
    """Raised when a requested resource is not found."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            additional_info=additional_info
        )

class DatabaseError(BaseAppException):
    """Raised when database operations fail."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            additional_info=additional_info
        )

class ExternalServiceError(BaseAppException):
    """Raised when external service calls fail."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status.HTTP_502_BAD_GATEWAY,
            additional_info=additional_info
        )

class CacheError(BaseAppException):
    """Raised when cache operations fail."""
    def __init__(self, detail: str, additional_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            error_code="CACHE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            additional_info=additional_info
        ) 