"""Shared error handling for knowledge agents."""

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