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
        
    def log_error(self, logger):
        """Log the error with structured information."""
        error_info = {
            "error_code": self.error_code,
            "status_code": self.status_code,
            "message": self.message
        }
        if self.details:
            error_info["details"] = self.details
        if self.original_error:
            error_info["original_error"] = str(self.original_error)
            error_info["original_error_type"] = type(self.original_error).__name__
        
        logger.error(f"API Error: {self.error_code}", extra=error_info)
        
    def to_dict(self):
        """Convert the error to a dictionary for API responses."""
        result = {
            "message": self.message,
            "error_code": self.error_code
        }
        if self.details:
            result["details"] = self.details
        return result


class ProcessingError(Exception):
    """Base error class for processing errors."""
    
    def __init__(
        self,
        message: str,
        operation: str = None,
        resource: str = None,
        original_error: Exception = None
    ):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.resource = resource
        self.original_error = original_error
        self.status_code = 500  # Default status code for processing errors
        
    def __str__(self):
        error_parts = [self.message]
        if self.operation:
            error_parts.append(f"Operation: {self.operation}")
        if self.resource:
            error_parts.append(f"Resource: {self.resource}")
        if self.original_error:
            error_parts.append(f"Original error: {str(self.original_error)}")
        return " | ".join(error_parts)
    
    def log_error(self, logger):
        """Log the error details using the provided logger."""
        error_message = str(self)
        logger.error(error_message)
        if self.original_error:
            logger.error(f"Original error details: {self.original_error.__class__.__name__}")
            
    def to_dict(self):
        """Convert the error to a dictionary for API responses."""
        return {
            "error": self.message,
            "operation": self.operation,
            "resource": self.resource,
            "type": self.__class__.__name__
        }


class ConfigurationError(APIError):
    """Error raised when there's an issue with configuration settings."""
    
    def __init__(
        self,
        message: str,
        config_key: str = None,
        config_value: Any = None,
        original_error: Exception = None
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)
            
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details,
            original_error=original_error
        )
        self.config_key = config_key
        self.config_value = config_value
        
    def log_error(self, logger):
        """Log the configuration error with additional context."""
        error_info = {
            "error_code": self.error_code,
            "status_code": self.status_code,
            "message": self.message,
            "config_key": self.config_key,
            "config_value": str(self.config_value) if self.config_value is not None else None
        }
        
        if self.original_error:
            error_info["original_error"] = str(self.original_error)
            error_info["original_error_type"] = type(self.original_error).__name__
            
        logger.error(f"Configuration Error: {self.config_key}", extra=error_info)


class ValidationError(APIError):
    """Error raised when there's an issue with request validation."""
    
    def __init__(
        self,
        message: str = None,
        field: str = None,
        value: Any = None,
        detail: str = None,
        status_code: int = 400
    ):
        # Use either message or detail
        error_message = message or detail or "Validation error"
        
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
            
        super().__init__(
            message=error_message,
            status_code=status_code,
            error_code="VALIDATION_ERROR",
            details=details
        )
        self.field = field
        self.value = value
        
    def log_error(self, logger):
        """Log the validation error with additional context."""
        error_info = {
            "error_details": {  # Nest the error info to avoid conflicts
                "error_code": self.error_code,
                "status_code": self.status_code,
                "field": self.field,
                "value": str(self.value) if self.value is not None else None
            }
        }
            
        logger.error(
            f"Validation Error: {self.field or 'unknown field'} - {self.message}", 
            extra=error_info
        ) 