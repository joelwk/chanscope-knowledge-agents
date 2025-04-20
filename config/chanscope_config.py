"""
Configuration for the Chanscope data management system.

This module provides configuration classes for the Chanscope approach,
supporting both file-based storage (Docker) and database storage (Replit).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from config.env_loader import detect_environment

class ChanScopeConfig:
    """
    Configuration for the Chanscope data management system.
    
    This class provides a unified configuration interface for both
    file-based (Docker) and database (Replit) storage implementations.
    """
    
    def __init__(
        self,
        root_data_path: Path,
        stratified_data_path: Optional[Path] = None,
        temp_path: Optional[Path] = None,
        filter_date: Optional[str] = None,
        sample_size: int = 100000,
        time_column: str = 'posted_date_time',
        strata_column: Optional[str] = None,
        embedding_batch_size: int = 10,
        env: Optional[str] = None,
        force_refresh: bool = False
    ):
        """
        Initialize the configuration.
        
        Args:
            root_data_path: Path to the data directory
            stratified_data_path: Path to the stratified sample directory
            temp_path: Path to the temporary files directory
            filter_date: Filter date for data (format: YYYY-MM-DD)
            sample_size: Size of stratified sample
            time_column: Name of time column for stratification
            strata_column: Name of strata column for stratification
            embedding_batch_size: Batch size for embedding generation
            env: Environment type ('docker' or 'replit')
            force_refresh: Whether to force refresh all data
        """
        self.root_data_path = root_data_path
        self.stratified_data_path = stratified_data_path or root_data_path / 'stratified'
        self.temp_path = temp_path or Path('temp_files')
        self.filter_date = filter_date
        self.sample_size = sample_size
        self.time_column = time_column
        self.strata_column = strata_column
        self.embedding_batch_size = embedding_batch_size
        self.force_refresh = force_refresh
        
        # Determine environment
        self.env = env or detect_environment()
    
    def _detect_environment(self) -> str:
        """Detect the execution environment. 
        DEPRECATED: Use config.env_loader.detect_environment() directly instead."""
        return detect_environment()
    
    @classmethod
    def from_env(cls, env_override: Optional[str] = None) -> 'ChanScopeConfig':
        """
        Create a configuration from environment variables.
        
        Args:
            env_override: Override for environment detection
            
        Returns:
            ChanScopeConfig: Configuration for the detected environment
        """
        env = env_override or detect_environment()
        
        if env == 'replit':
            # Replit configuration
            repl_home = os.environ.get('REPL_HOME', os.getcwd())
            root_data_path = Path(os.environ.get('ROOT_DATA_PATH', os.path.join(repl_home, 'data')))
            stratified_data_path = Path(os.environ.get('STRATIFIED_PATH', os.path.join(repl_home, 'data/stratified')))
            temp_path = Path(os.environ.get('PATH_TEMP', os.path.join(repl_home, 'temp_files')))
        else:
            # Docker/local configuration
            root_data_path = Path(os.environ.get('ROOT_DATA_PATH', 'data'))
            stratified_data_path = Path(os.environ.get('STRATIFIED_PATH', 'data/stratified'))
            temp_path = Path(os.environ.get('PATH_TEMP', 'temp_files'))
        
        # Common configuration
        filter_date = os.environ.get('FILTER_DATE')
        sample_size = int(os.environ.get('SAMPLE_SIZE', '10000'))
        time_column = os.environ.get('TIME_COLUMN', 'posted_date_time')
        strata_column = os.environ.get('STRATA_COLUMN')
        embedding_batch_size = int(os.environ.get('EMBEDDING_BATCH_SIZE', '10'))
        force_refresh = os.environ.get('FORCE_DATA_REFRESH', '').lower() in ('true', '1', 'yes')
        
        return cls(
            root_data_path=root_data_path,
            stratified_data_path=stratified_data_path,
            temp_path=temp_path,
            filter_date=filter_date,
            sample_size=sample_size,
            time_column=time_column,
            strata_column=strata_column,
            embedding_batch_size=embedding_batch_size,
            env=env,
            force_refresh=force_refresh
        )
    
    def get_env_specific_attributes(self) -> Dict[str, Any]:
        """
        Get environment-specific attributes.
        
        Returns:
            Dict[str, Any]: Dictionary of environment-specific attributes
        """
        if self.env == 'replit':
            return {
                'use_replit_db': True,
                'use_object_storage': True,  # Flag to use Replit Object Storage for embeddings
                'database_url': os.environ.get('DATABASE_URL', ''),
                'pghost': os.environ.get('PGHOST', ''),
                'pguser': os.environ.get('PGUSER', ''),
                'pgpassword': os.environ.get('PGPASSWORD', ''),
                'replit_db_url': os.environ.get('REPLIT_DB_URL', '')
            }
        else:
            return {
                'use_replit_db': False,
                'use_object_storage': False,
                'complete_data_file': self.root_data_path / 'complete_data.csv',
                'stratified_file': self.stratified_data_path / 'stratified_sample.csv',
                'embeddings_file': self.stratified_data_path / 'embeddings.npz',
                'thread_id_map_file': self.stratified_data_path / 'thread_id_map.json'
            }
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        env_attrs = self.get_env_specific_attributes()
        
        return (
            f"ChanScopeConfig(env={self.env}, "
            f"root_data_path={self.root_data_path}, "
            f"stratified_data_path={self.stratified_data_path}, "
            f"temp_path={self.temp_path}, "
            f"filter_date={self.filter_date}, "
            f"sample_size={self.sample_size}, "
            f"time_column={self.time_column}, "
            f"embedding_batch_size={self.embedding_batch_size}, "
            f"force_refresh={self.force_refresh}, "
            f"env_specific={env_attrs})"
        ) 