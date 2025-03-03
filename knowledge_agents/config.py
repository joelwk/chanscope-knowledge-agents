"""
Configuration classes for the knowledge agents.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from config.settings import Config
from config.base_settings import get_base_settings

@dataclass
class ChanScopeConfig:
    """Configuration for the Chanscope approach."""
    
    # Data paths
    root_data_path: Path
    stratified_data_path: Path
    temp_path: Path
    
    # File names
    complete_data_file: str = "complete_data.csv"
    stratified_file: str = "stratified_sample.csv"
    embeddings_file: str = "embeddings.npz"
    thread_id_map_file: str = "thread_id_map.json"
    
    # Stratification settings
    stratify_sample_size: int = 100000
    stratify_random_seed: int = 42
    
    # Update settings
    force_refresh: bool = False
    update_marker_timeout: int = 300  # seconds
    
    @classmethod
    def from_config(cls, config: Optional[Config] = None) -> 'ChanScopeConfig':
        """Create a ChanScopeConfig from a Config object."""
        if config is None:
            config = get_base_settings()
            
        # Set up paths
        root_data_path = Path(config.DATA_DIR)
        stratified_data_path = root_data_path / "stratified"
        temp_path = Path(config.TEMP_DIR)
        
        return cls(
            root_data_path=root_data_path,
            stratified_data_path=stratified_data_path,
            temp_path=temp_path
        ) 