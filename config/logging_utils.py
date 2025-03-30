"""Utilities for logging configuration and sensitive data handling."""
import logging
import re
from typing import Any, Dict, Optional, Union

# Define a list of sensitive keys that should be obfuscated in logs
SENSITIVE_KEYS = [
    'password', 'secret', 'key', 'token', 'auth', 'credential', 'pgpassword', 'aws_secret',
    'apikey', 'api_key', 'access_key', 'replit_db_url', 'database_url', 'connection_string'
]

def obfuscate_sensitive_data(data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
    """
    Obfuscate sensitive data in strings or dictionaries.
    
    Args:
        data: String or dictionary containing potentially sensitive data
        
    Returns:
        Obfuscated version of the input
    """
    if isinstance(data, dict):
        # Handle dictionary data (like the extra parameter for logging)
        result = {}
        for key, value in data.items():
            # Check if the key contains any sensitive patterns
            if any(pattern.lower() in key.lower() for pattern in SENSITIVE_KEYS):
                if isinstance(value, str) and value:
                    # Show only first 3 and last 3 characters if string is long enough
                    if len(value) > 8:
                        result[key] = f"{value[:3]}...{value[-3:]}"
                    else:
                        result[key] = "***"
                else:
                    result[key] = "***"
            else:
                # Recursively process values that might be dictionaries
                if isinstance(value, dict):
                    result[key] = obfuscate_sensitive_data(value)
                else:
                    result[key] = value
        return result
    elif isinstance(data, str):
        # Handle string data using regex patterns
        # Match patterns like 'password=xyz123', 'api_key: "abc987"', etc.
        for pattern in SENSITIVE_KEYS:
            # Regular expressions to find sensitive data patterns
            regex_patterns = [
                rf'{pattern}[=:]\s*["\']?([^"\']+)["\']?',  # key=value, key: value
                rf'{pattern}[=:]\s*["\']([^"\']+)["\']',    # key="value"
                rf'"?{pattern}"?\s*:\s*"([^"]+)"',          # "key": "value"
                rf"'?{pattern}'?\s*:\s*'([^']+)'"           # 'key': 'value'
            ]
            
            for regex in regex_patterns:
                data = re.sub(
                    regex,
                    lambda m: m.group(0).replace(m.group(1), "***"),
                    data,
                    flags=re.IGNORECASE
                )
        return data
    else:
        # For other types, just return as is
        return data

class SensitiveFilter(logging.Filter):
    """Filter that obfuscates sensitive information in log records."""
    
    def filter(self, record):
        """Process the log record to obfuscate sensitive information."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = obfuscate_sensitive_data(record.msg)
            
        # Handle extra data passed via keyword arguments
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = obfuscate_sensitive_data(record.args)
            elif isinstance(record.args, (list, tuple)):
                # Convert args tuple to list, process items that are strings, then back to tuple
                args_list = list(record.args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, str):
                        args_list[i] = obfuscate_sensitive_data(arg)
                    elif isinstance(arg, dict):
                        args_list[i] = obfuscate_sensitive_data(arg)
                record.args = tuple(args_list)
        
        return True

def setup_sensitive_data_filter():
    """Add the sensitive data filter to all existing loggers."""
    sensitive_filter = SensitiveFilter()
    
    # Add filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(sensitive_filter)
    
    # Add filters to all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.addFilter(sensitive_filter)
            
    # Also add to specific important loggers
    for logger_name in ['api', 'config', 'knowledge_agents', 'uvicorn', 'fastapi']:
        logger = logging.getLogger(logger_name)
        logger.addFilter(sensitive_filter)
        
    return sensitive_filter 