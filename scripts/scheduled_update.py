#!/usr/bin/env python
"""
Unified Scheduled Data Update Script for the Chanscope Application

This script handles scheduled data updates for the Chanscope application
with unified support for:
- Docker environments (file-based storage)
- Replit environments (database storage)
- Local development environments

Features:
- Auto-detection of environment
- Support for one-time and continuous updates
- Comprehensive data status reporting
- Robust error handling with auto-recovery

Usage:
  python scripts/scheduled_update.py refresh  # Update all data
  python scripts/scheduled_update.py status   # Check data status
  python scripts/scheduled_update.py embeddings  # Generate embeddings only
  
  # Continuous updates (scheduler mode)
  python scripts/scheduled_update.py refresh --continuous --interval=3600
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import traceback
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Add the project root to the Python path to import from packages
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Import environment utilities
from config.env_loader import load_environment, is_replit_environment
from config.logging_config import setup_logging

# Import the ChanScope manager
from knowledge_agents.data_processing.chanscope_manager import ChanScopeDataManager
from config.chanscope_config import ChanScopeConfig

# Set up logging
setup_logging()
logger = logging.getLogger("scheduled_update")

# Initialize environment variables
load_environment()

class ScheduledUpdater:
    """Main class for handling scheduled data updates"""
    
    def __init__(self, args):
        """Initialize the updater with command line arguments"""
        self.args = args
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.environment = self._detect_environment()
        self.config = self._create_config()
        self.data_manager = None
        self.initialize_data_manager()
        
    def _detect_environment(self) -> str:
        """Detect the current execution environment"""
        # Use provided environment if specified
        if self.args.env:
            logger.info(f"Using specified environment: {self.args.env}")
            return self.args.env
            
        # Auto-detect environment
        if is_replit_environment():
            logger.info("Auto-detected Replit environment")
            return 'replit'
        elif os.path.exists('/.dockerenv') or any('docker' in line for line in open('/proc/1/cgroup', 'r').readlines() if 'docker' in line):
            logger.info("Auto-detected Docker environment")
            return 'docker'
        else:
            logger.info("No specific environment detected, using local environment")
            return 'local'
    
    def _create_config(self) -> ChanScopeConfig:
        """Create ChanScopeConfig with environment-specific settings"""
        config = ChanScopeConfig.from_env(env_override=self.environment)
        
        # Override config with command-line arguments
        if self.args.force_refresh:
            config.force_refresh = True
        if self.args.filter_date:
            config.filter_date = self.args.filter_date
            
        # Set any environment-specific optimizations
        if self.environment == 'replit':
            # Use smaller batch sizes and processing chunks for Replit
            config.embedding_batch_size = int(os.environ.get('EMBEDDING_BATCH_SIZE', '5'))
            config.embedding_chunk_size = int(os.environ.get('PROCESSING_CHUNK_SIZE', '1000'))
            config.max_workers = 1  # Replit has limited resources
            
        logger.info(f"Using configuration: {config}")
        return config
    
    def initialize_data_manager(self):
        """Initialize the ChanScopeDataManager"""
        try:
            self.data_manager = ChanScopeDataManager(self.config)
            logger.info("ChanScopeDataManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChanScopeDataManager: {e}")
            logger.error(traceback.format_exc())
            raise
            
    async def run_data_refresh(self) -> bool:
        """Run full data refresh following Chanscope approach"""
        try:
            logger.info("Starting full data refresh")
            logger.info(f"Config: force_refresh={self.config.force_refresh}, filter_date={self.config.filter_date}")
            
            if self.args.force_refresh:
                logger.info("Force refresh flag explicitly set via command line argument")
                self.config.force_refresh = True
            
            if self.config.force_refresh:
                logger.info("Will regenerate stratified sample and embeddings")
            elif not self.config.force_refresh:
                logger.info("Note: force_refresh is False - existing stratified sample may be used if available")
                logger.info("To force regeneration, run with: --force-refresh")
            
            if self.config.filter_date is None:
                logger.info("No explicit filter_date provided - database query will use retention_days from settings")
                from config.base_settings import get_base_settings
                base_settings = get_base_settings()
                processing = base_settings.get('processing', {})
                retention_days = processing.get('retention_days', 30)
                logger.info(f"Expected retention period: {retention_days} days")
            
            # Create Chanscope data manager
            data_manager = ChanScopeDataManager(self.config)
            
            # Create a marker file to indicate update in progress
            marker_path = Path(self.config.root_data_path) / ".update_in_progress"
            with open(marker_path, 'w') as f:
                f.write(f"Update started at {datetime.now().isoformat()}")
            
            # Check current data status
            row_count = await data_manager.complete_data_storage.get_row_count()
            logger.info(f"Current complete data row count: {row_count}")
            
            # Force refresh for empty database
            if row_count == 0 and not self.config.force_refresh:
                logger.info("Database is empty, enabling force_refresh")
                self.config.force_refresh = True
            
            # Ensure data is ready
            success = await data_manager.ensure_data_ready(
                force_refresh=self.config.force_refresh,
                skip_embeddings=self.args.skip_embeddings
            )
            
            # Remove marker file
            if marker_path.exists():
                marker_path.unlink()
            
            if success:
                logger.info("Data refresh completed successfully")
                await self._update_status_file(success=True)
                return True
            else:
                logger.error("Data refresh failed")
                await self._update_status_file(success=False, error="Data refresh operation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during data refresh: {e}")
            logger.error(traceback.format_exc())
            
            # Remove marker file if it exists
            if marker_path.exists():
                marker_path.unlink()
                
            await self._update_status_file(success=False, error=str(e))
            return False
    
    async def generate_embeddings(self) -> bool:
        """Generate embeddings only"""
        logger.info(f"Generating embeddings (force_refresh={self.config.force_refresh})")
        
        try:
            # Create a marker file
            marker_path = Path(self.config.root_data_path) / ".embeddings_in_progress"
            with open(marker_path, 'w') as f:
                f.write(f"Embedding generation started at {datetime.now().isoformat()}")
            
            # Generate embeddings
            success = await self.data_manager.generate_embeddings(force_refresh=self.config.force_refresh)
            
            # Remove marker file
            if marker_path.exists():
                marker_path.unlink()
            
            if success:
                logger.info("Embedding generation completed successfully")
                await self._update_status_file(success=True, operation="embeddings")
                return True
            else:
                logger.error("Embedding generation failed")
                await self._update_status_file(success=False, error="Embedding generation failed", operation="embeddings")
                return False
                
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            logger.error(traceback.format_exc())
            
            # Remove marker file if it exists
            if marker_path.exists():
                marker_path.unlink()
                
            await self._update_status_file(success=False, error=str(e), operation="embeddings")
            return False
    
    async def check_data_status(self) -> Dict[str, Any]:
        """Check current data status"""
        logger.info("Checking data status")
        
        try:
            # Get row count
            row_count = 0
            try:
                row_count = await self.data_manager.complete_data_storage.get_row_count()
            except Exception as e:
                logger.warning(f"Could not get row count: {e}")
            
            # Check if stratified sample exists
            stratified_exists = False
            try:
                stratified_exists = await self.data_manager.stratified_storage.sample_exists()
            except Exception as e:
                logger.warning(f"Could not check stratified sample: {e}")
            
            # Check if embeddings exist
            embeddings_exist = False
            try:
                embeddings_exist = await self.data_manager.embedding_storage.embeddings_exist()
            except Exception as e:
                logger.warning(f"Could not check embeddings: {e}")
            
            # Get processing state
            state = None
            try:
                state = await self.data_manager.state_manager.get_state()
            except Exception as e:
                logger.warning(f"Could not get processing state: {e}")
            
            # Check if data is ready
            data_ready = await self.data_manager.is_data_ready(skip_embeddings=False)
            
            # Compile status report
            status = {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "complete_data": {
                    "exists": row_count > 0,
                    "row_count": row_count
                },
                "stratified_sample": {
                    "exists": stratified_exists
                },
                "embeddings": {
                    "exists": embeddings_exist
                },
                "processing_state": state,
                "data_ready": data_ready,
                "config": {
                    "force_refresh": self.config.force_refresh,
                    "filter_date": self.config.filter_date,
                    "sample_size": self.config.sample_size
                }
            }
            
            # Log status summary
            logger.info(f"Complete data row count: {row_count}")
            logger.info(f"Stratified sample exists: {stratified_exists}")
            logger.info(f"Embeddings exist: {embeddings_exist}")
            logger.info(f"Data is ready: {data_ready}")
            
            # Write status to file
            status_path = Path(self.config.root_data_path) / "data_status.json"
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
                
            logger.info(f"Status saved to {status_path}")
            return status
            
        except Exception as e:
            logger.error(f"Error checking data status: {e}")
            logger.error(traceback.format_exc())
            return {
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "error": str(e)
            }
    
    async def _update_status_file(self, success: bool, error: Optional[str] = None, 
                                 operation: str = "refresh") -> None:
        """Update the status file with the result of the operation"""
        try:
            status_path = Path(self.config.root_data_path) / "last_update_status.json"
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "success": success,
                "environment": self.environment
            }
            
            if error:
                status["error"] = error
                
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
                
            logger.info(f"Updated status file: {status_path}")
        except Exception as e:
            logger.warning(f"Could not update status file: {e}")

    async def run_continuous_updates(self) -> None:
        """Run continuous updates at the specified interval"""
        interval = self.args.interval
        logger.info(f"Starting continuous updates with interval {interval} seconds")
        
        while True:
            try:
                if self.args.command == "refresh":
                    await self.run_data_refresh()
                elif self.args.command == "embeddings":
                    await self.generate_embeddings()
                elif self.args.command == "status":
                    await self.check_data_status()
                else:
                    logger.error(f"Unknown command for continuous mode: {self.args.command}")
                    
                logger.info(f"Waiting {interval} seconds until next update...")
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in update cycle: {e}")
                logger.error(traceback.format_exc())
                # Wait a bit before retrying to avoid rapid failure loops
                await asyncio.sleep(60)
                
                # Re-initialize data manager in case of connection issues
                try:
                    self.initialize_data_manager()
                except Exception as reinit_error:
                    logger.error(f"Failed to re-initialize data manager: {reinit_error}")

async def run_update(args):
    """Main function to run the update based on command-line arguments"""
    updater = ScheduledUpdater(args)
    
    if args.command == "refresh":
        return await updater.run_data_refresh()
    elif args.command == "embeddings":
        return await updater.generate_embeddings()
    elif args.command == "status":
        await updater.check_data_status()
        return True
    else:
        logger.error(f"Unknown command: {args.command}")
        return False

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified scheduled data updates for Chanscope',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Command to run
    parser.add_argument('command', choices=['refresh', 'embeddings', 'status'], 
                        help='Command to run (refresh=update all data, embeddings=generate embeddings, status=check status)')
    
    # Run mode
    parser.add_argument('--continuous', action='store_true',
                        help='Run in continuous mode with periodic updates')
    parser.add_argument('--run-once', action='store_true',
                        help='Run once and exit (default behavior, for backward compatibility)')
    parser.add_argument('--interval', type=int, default=3600,
                        help='Interval in seconds between updates when running in continuous mode (default: 3600)')
    
    # Configuration options
    parser.add_argument('--env', choices=['docker', 'replit'],
                        help='Override environment detection')
    parser.add_argument('--filter-date', type=str,
                        help='Filter data based on date (ISO format)')
    
    # Update options
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force recreation of stratified sample and embeddings even if they exist. Important: Without this flag, existing stratified samples may be used even if outdated.')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation when refreshing data')
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_args()
    
    # Handle run-once flag (for backward compatibility)
    if args.run_once:
        args.continuous = False
    
    try:
        # Run the update
        if args.continuous:
            updater = ScheduledUpdater(args)
            await updater.run_continuous_updates()
            return 0
        else:
            success = await run_update(args)
            return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Update process interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unhandled error in update process: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)