"""Utility functions for configuration handling and conversion.

This module serves as the central location for all configuration-related utilities,
including request validation, configuration building, and date handling.

The module is organized into several key areas:
1. Request Models - Pydantic models for API requests
2. Date Utilities - Date parsing and validation functions
3. Configuration Validation - Functions to validate various config types
4. Configuration Building - Functions to construct config objects
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pandas as pd
from knowledge_agents.model_ops import ModelConfig
from pydantic import BaseModel, Field
import os
from config.logging_config import get_logger
from config.base import BaseConfig, get_base_settings

# Configure logging
logger = get_logger(__name__)

# Default settings
DEFAULT_SETTINGS = get_base_settings()

# ============================================================================
# Request Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query processing with configuration options.
    
    This model defines the structure for incoming query requests, including
    all possible configuration parameters that can be specified by the client.
    """
    query: str = Field(..., description="The query to process")
    force_refresh: bool = Field(False, description="Force refresh the data cache")
    skip_embeddings: bool = Field(False, description="Skip embedding generation")
    skip_batching: bool = Field(False, description="Skip batching")
    skip_cache: bool = Field(False, description="Skip cache lookup for query")
    filter_date: Optional[str] = Field(None, description="Filter results by date (format: YYYY-MM-DD HH:MM:SS+00:00)")
    select_board: Optional[str] = Field(None, description="Filter S3 data by board ID")
    # Configuration options
    sample_size: Optional[int] = Field(None, description="Sample size for processing")
    embedding_batch_size: Optional[int] = Field(None, description="Batch size for embedding operations")
    chunk_batch_size: Optional[int] = Field(None, description="Batch size for chunk generation")
    summary_batch_size: Optional[int] = Field(None, description="Batch size for summary generation")
    max_workers: Optional[int] = Field(None, description="Maximum number of workers for parallel processing")
    # Provider options
    embedding_provider: Optional[str] = Field(None, description="Provider for embeddings (openai/grok/venice)")
    chunk_provider: Optional[str] = Field(None, description="Provider for chunk generation (openai/grok/venice)")
    summary_provider: Optional[str] = Field(None, description="Provider for summarization (openai/grok/venice)")
    use_background: bool = Field(False, description="Use background processing")

class QueryResponse(BaseModel):
    """Response model for query processing.
    
    This model defines the structure for query processing responses,
    including the generated chunks and summary.
    """
    chunks: List[Dict[str, Any]] = Field(..., description="List of processed chunks with metadata")
    summary: str = Field(..., description="Generated summary of the chunks")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata about the processing")

class BatchQueryRequest(BaseModel):
    """Request model for batch query processing.
    
    This model allows multiple queries to be processed with shared configuration.
    """
    queries: List[str] = Field(..., description="List of queries to process")
    config: Optional[QueryRequest] = Field(None, description="Configuration options for all queries")
    
    # Batch processing parameters
    chunk_batch_size: Optional[int] = Field(None, description="Batch size for chunk generation")
    summary_batch_size: Optional[int] = Field(None, description="Batch size for summary generation")
    max_workers: Optional[int] = Field(None, description="Maximum number of workers for parallel processing")
    
    # Skip cache option
    skip_cache: bool = Field(False, description="Skip cache lookup for queries")

class QueryValidationResult(BaseModel):
    """Validation result for query requests.
    
    Provides structured feedback about query validation, including any error messages
    and formatted dates.
    """
    is_valid: bool
    error_message: Optional[str] = None
    formatted_date: Optional[str] = None

# ============================================================================
# Date Utilities
# ============================================================================

def parse_filter_date(date_str: Optional[str]) -> Optional[str]:
    """Parse and validate filter date string."""
    if not date_str:
        return None
        
    try:
        # Parse with UTC awareness
        date = pd.to_datetime(date_str, utc=True)
        if date.tzinfo is None:
            date = date.tz_localize('UTC')
        return date.strftime('%Y-%m-%d')
    except Exception as e:
        logger.warning(f"Error parsing date {date_str}: {e}")
        return None

def get_filter_date(runtime_override: Optional[str] = None) -> Optional[str]:
    """Get filter date with request-local override support."""
    if runtime_override is not None:
        return parse_filter_date(runtime_override)
    
    env_date = DEFAULT_SETTINGS['processing']['filter_date']
    return parse_filter_date(env_date) if env_date else None

def validate_query_request(
    query: str,
    filter_date: Optional[str] = None,
    force_refresh: bool = False
) -> QueryValidationResult:
    """
    Validate a query request parameters.
    
    Args:
        query: The query string to validate
        filter_date: Optional filter date to validate
        force_refresh: Whether to force refresh the data cache
        
    Returns:
        QueryValidationResult with validation status and any error messages
    """
    if not query or not query.strip():
        return QueryValidationResult(
            is_valid=False,
            error_message="Query cannot be empty"
        )
    
    if len(query) > 1000:  # Reasonable limit for query length
        return QueryValidationResult(
            is_valid=False,
            error_message="Query exceeds maximum length of 1000 characters"
        )
    
    if filter_date:
        try:
            formatted_date = parse_filter_date(filter_date)
            if formatted_date is None:
                return QueryValidationResult(
                    is_valid=False,
                    error_message="Failed to parse filter_date"
                )
            return QueryValidationResult(
                is_valid=True,
                formatted_date=formatted_date
            )
        except ValueError as e:
            return QueryValidationResult(
                is_valid=False,
                error_message=str(e)
            )
    
    return QueryValidationResult(is_valid=True)

# ============================================================================
# Configuration Validation
# ============================================================================

def validate_paths(model_config: ModelConfig) -> None:
    """Validate path configurations and ensure they exist.
    
    This function validates all required paths in the configuration and creates
    directories if they don't exist. It's a critical step in ensuring the
    application has proper access to all necessary filesystem locations.
    
    Args:
        model_config: ModelConfig instance to validate
        
    Raises:
        ValueError: If any required paths are missing or invalid
    """
    required_paths = [
        ('root_data_path', model_config.root_data_path),
        ('stratified_path', model_config.stratified_path),
        ('temp_path', model_config.temp_path)
    ]
    
    for path_name, path_value in required_paths:
        if not path_value:
            raise ValueError(f"Missing required path: {path_name}")
        
        path = Path(path_value)
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Validated directory path: {path}")
        except Exception as e:
            raise ValueError(f"Failed to create/validate directory {path_name}: {e}")

def validate_model_config(model_settings: Union[Dict[str, Any], ModelConfig]) -> Dict[str, Any]:
    """Validate model configuration settings."""
    validated = {}
    
    # Convert ModelConfig to dict if needed
    if isinstance(model_settings, ModelConfig):
        model_settings = model_settings.model_settings
    
    # Validate batch sizes
    for key in ['embedding_batch_size', 'chunk_batch_size', 'summary_batch_size']:
        value = model_settings.get(key, 0)
        try:
            value = int(value)
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            validated[key] = value
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid {key}: {value}")
            raise ValueError(f"Invalid {key}: {value}") from e
            
    # Validate provider settings
    for key in ['default_embedding_provider', 'default_chunk_provider', 'default_summary_provider']:
        value = model_settings.get(key)
        if not value:
            raise ValueError(f"Missing required setting: {key}")
        if value not in ['openai', 'venice', 'grok']:
            raise ValueError(f"Invalid provider for {key}: {value}")
        validated[key] = value
            
    return validated

def build_model_config(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build model configuration from settings or environment."""
    if settings is None:
        settings = BaseConfig.get_base_settings().get('model', {})
    
    return validate_model_config(settings)

def build_data_config(model_config: ModelConfig) -> 'DataConfig':
    """Build data configuration from model configuration.
    
    This function constructs a DataConfig instance using settings from
    the model configuration. It ensures all necessary paths and settings
    are properly initialized.
    
    Args:
        model_config: ModelConfig instance containing base settings
        
    Returns:
        DataConfig: Configured data operations instance
    """
    from knowledge_agents.data_ops import DataConfig  # Import here to avoid circular dependency
    
    return DataConfig(
        root_data_path=model_config.root_data_path,
        stratified_data_path=model_config.stratified_path,
        temp_path=model_config.temp_path,
        filter_date=model_config.filter_date,
        sample_size=model_config.sample_size
    )

def build_config_from_request(request: QueryRequest) -> Dict[str, Any]:
    """Build configuration from query request parameters.
    
    Args:
        request: The query request with configuration parameters
        
    Returns:
        Configuration dictionary with validated parameters
    """
    # Get base settings as the starting point
    config = get_base_settings()
    processing = config.get('processing', {})
    
    # Apply overrides from request if set
    if request.filter_date is not None:
        processing['filter_date'] = parse_filter_date(request.filter_date)
    
    if request.select_board is not None:
        processing['select_board'] = request.select_board
    
    # Apply sample size if valid
    if request.sample_size is not None:
        sample_settings = config.get('sample', {})
        min_samples = sample_settings.get('min_sample_size', 100)
        max_samples = sample_settings.get('max_sample_size', 10000)
        processing['sample_size'] = max(min_samples, min(request.sample_size, max_samples))
    
    # Apply batch sizes
    chunk_settings = config.get('chunk', {})
    if request.embedding_batch_size is not None:
        chunk_settings['embedding_batch_size'] = request.embedding_batch_size
    if request.chunk_batch_size is not None:
        chunk_settings['chunk_batch_size'] = request.chunk_batch_size
    if request.summary_batch_size is not None:
        chunk_settings['summary_batch_size'] = request.summary_batch_size
    
    # Apply worker limit
    if request.max_workers is not None:
        processing['max_workers'] = request.max_workers
    
    # Apply provider selection
    if request.embedding_provider is not None:
        config.setdefault('providers', {})['embedding'] = request.embedding_provider
    if request.chunk_provider is not None:
        config.setdefault('providers', {})['chunk'] = request.chunk_provider
    if request.summary_provider is not None:
        config.setdefault('providers', {})['summary'] = request.summary_provider
    
    # Update config with new settings
    config['processing'] = processing
    config['chunk'] = chunk_settings
    
    return config

def build_unified_config(request: Union[Dict[str, Any], QueryRequest]) -> ModelConfig:
    """Build a unified ModelConfig from a configuration dictionary or QueryRequest."""
    # Initialize with base settings
    base_settings = BaseConfig.get_base_settings()
    config_dict = {
        'model': base_settings.get('model', {}),
        'processing': base_settings.get('processing', {}),
        'paths': base_settings.get('paths', {})
    }
    
    if isinstance(request, QueryRequest):
        # Update model settings from request
        if request.embedding_provider:
            config_dict['model']['default_embedding_provider'] = request.embedding_provider
        if request.chunk_provider:
            config_dict['model']['default_chunk_provider'] = request.chunk_provider
        if request.summary_provider:
            config_dict['model']['default_summary_provider'] = request.summary_provider
        if request.embedding_batch_size:
            config_dict['model']['embedding_batch_size'] = request.embedding_batch_size
        if request.chunk_batch_size:
            config_dict['model']['chunk_batch_size'] = request.chunk_batch_size
        if request.summary_batch_size:
            config_dict['model']['summary_batch_size'] = request.summary_batch_size
            
        # Update processing settings from request
        if request.filter_date:
            parsed_date = parse_filter_date(request.filter_date)
            if parsed_date:
                config_dict['processing']['filter_date'] = parsed_date
                logger.info(f"Using parsed filter date: {parsed_date}")
            else:
                logger.warning(f"Failed to parse filter date: {request.filter_date}")
        if request.sample_size:
            config_dict['processing']['sample_size'] = request.sample_size
        if request.max_workers:
            config_dict['processing']['max_workers'] = request.max_workers
    else:
        # If it's a dictionary, merge it with base settings
        config_dict['model'].update(request.get('model', {}))
        config_dict['processing'].update(request.get('processing', {}))
        config_dict['paths'].update(request.get('paths', {}))
        
        # Parse filter date if present in processing settings
        if 'filter_date' in config_dict['processing']:
            parsed_date = parse_filter_date(config_dict['processing']['filter_date'])
            if parsed_date:
                config_dict['processing']['filter_date'] = parsed_date
                logger.info(f"Using parsed filter date: {parsed_date}")
            else:
                logger.warning(f"Failed to parse filter date: {config_dict['processing']['filter_date']}")
    
    # Validate model settings
    model_settings = validate_model_config(config_dict['model'])
    
    # Create ModelConfig instance with the entire config dictionary
    return ModelConfig(**config_dict)