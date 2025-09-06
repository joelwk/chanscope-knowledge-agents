#!/usr/bin/env python
"""
Enhanced Automated Refresh System for ChanScope
Features:
- Auto-scheduling with configurable intervals
- Health monitoring and alerting
- Graceful error recovery
- Web dashboard for monitoring
"""

import os
import sys
import time
import asyncio
import logging
import json
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

# Add the project root to the Python path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/automated_refresh.log')
    ]
)
logger = logging.getLogger("automated_refresh")

class RefreshStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    SCHEDULED = "scheduled"

@dataclass
class RefreshJob:
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: RefreshStatus
    error_message: Optional[str] = None
    rows_processed: int = 0
    duration_seconds: float = 0

class AutomatedRefreshManager:
    """Enhanced refresh manager with monitoring and control"""
    
    def __init__(self, interval_seconds: int = 3600, max_retries: int = 3):
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries
        self.is_running = False
        self.current_job: Optional[RefreshJob] = None
        self.job_history: List[RefreshJob] = []
        self.status_file = Path("data/refresh_status.json")
        self.metrics_file = Path("data/refresh_metrics.json")
        self.current_row_count: int = 0
        self._setup_signal_handlers()
        
        # Create necessary directories
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
        if self.current_job:
            self.current_job.status = RefreshStatus.ERROR
            self.current_job.error_message = "Shutdown requested"
            self._save_status()
        sys.exit(0)
        
    def _save_status(self):
        """Save current status to file"""
        status_data = {
            "is_running": self.is_running,
            "current_job": asdict(self.current_job) if self.current_job else None,
            "next_run": (datetime.now() + timedelta(seconds=self.interval_seconds)).isoformat(),
            "interval_seconds": self.interval_seconds,
            "last_updated": datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        if status_data["current_job"]:
            for key in ["start_time", "end_time"]:
                if status_data["current_job"].get(key):
                    if isinstance(status_data["current_job"][key], datetime):
                        status_data["current_job"][key] = status_data["current_job"][key].isoformat()
            # Convert enum to string
            if "status" in status_data["current_job"]:
                status_data["current_job"]["status"] = status_data["current_job"]["status"].value
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
            
    def _save_metrics(self):
        """Save performance metrics"""
        if self.job_history:
            successful_jobs = [j for j in self.job_history if j.status == RefreshStatus.SUCCESS]
            failed_jobs = [j for j in self.job_history if j.status == RefreshStatus.ERROR]
            
            metrics = {
                "total_runs": len(self.job_history),
                "successful_runs": len(successful_jobs),
                "failed_runs": len(failed_jobs),
                "success_rate": len(successful_jobs) / len(self.job_history) * 100 if self.job_history else 0,
                "average_duration": sum(j.duration_seconds for j in successful_jobs) / len(successful_jobs) if successful_jobs else 0,
                "average_rows_processed": sum(j.rows_processed for j in successful_jobs) / len(successful_jobs) if successful_jobs else 0,
                "current_row_count": self.current_row_count,
                "last_success": max((j.end_time for j in successful_jobs), default=None),
                "last_failure": max((j.end_time for j in failed_jobs), default=None),
                "last_updated": datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            for key in ["last_success", "last_failure"]:
                if metrics.get(key) and isinstance(metrics[key], datetime):
                    metrics[key] = metrics[key].isoformat()
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
                
    async def run_refresh(self) -> RefreshJob:
        """Execute a single refresh job"""
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job = RefreshJob(
            id=job_id,
            start_time=datetime.now(),
            end_time=None,
            status=RefreshStatus.RUNNING,
            error_message=None,
            rows_processed=0
        )
        
        self.current_job = job
        self._save_status()

        try:
            logger.info(f"Starting refresh job {job_id}")

            # Import and run the scheduled update
            from scheduled_update import ScheduledUpdater
            import argparse

            # Create args for the updater
            args = argparse.Namespace(
                env=None,
                force_refresh=False,
                filter_date=None,
                skip_embeddings=False,
                continuous=False,
                interval=self.interval_seconds
            )

            updater = ScheduledUpdater(args)
            # Capture pre-refresh row count for delta metrics
            try:
                pre_status = await updater.check_data_status()
                pre_rows = pre_status.get("complete_data", {}).get("row_count", 0)
            except Exception:
                pre_rows = 0
            success = await updater.run_data_refresh()

            if success:
                job.status = RefreshStatus.SUCCESS
                logger.info(f"Refresh job {job_id} completed successfully")

                # Try to get row count
                try:
                    status = await updater.check_data_status()
                    post_rows = status.get("complete_data", {}).get("row_count", 0)
                    # rows_processed reflects delta for this job
                    job.rows_processed = max(0, int(post_rows) - int(pre_rows))
                    self.current_row_count = int(post_rows)
                except:
                    pass
            else:
                job.status = RefreshStatus.ERROR
                job.error_message = "Refresh operation failed"
                logger.error(f"Refresh job {job_id} failed")
                
        except Exception as e:
            job.status = RefreshStatus.ERROR
            job.error_message = str(e)
            logger.error(f"Error in refresh job {job_id}: {e}")
            
        finally:
            job.end_time = datetime.now()
            job.duration_seconds = (job.end_time - job.start_time).total_seconds()
            self.job_history.append(job)
            
            # Keep only last 100 jobs in history
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-100:]
                
            self.current_job = None
            self._save_status()
            self._save_metrics()
            
        return job
        
    async def run_continuous(self):
        """Run continuous refresh with retries and monitoring"""
        self.is_running = True
        consecutive_failures = 0
        
        logger.info(f"Starting automated refresh with interval: {self.interval_seconds} seconds")
        
        while self.is_running:
            try:
                # Run refresh
                job = await self.run_refresh()
                
                if job.status == RefreshStatus.SUCCESS:
                    consecutive_failures = 0
                    logger.info(f"Next refresh scheduled in {self.interval_seconds} seconds")
                else:
                    consecutive_failures += 1
                    logger.warning(f"Refresh failed ({consecutive_failures}/{self.max_retries})")
                    
                    if consecutive_failures >= self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) reached. Waiting longer before retry...")
                        await asyncio.sleep(self.interval_seconds * 2)  # Double wait time after max failures
                        consecutive_failures = 0  # Reset counter
                    else:
                        # Shorter wait for retries
                        await asyncio.sleep(60)  # 1 minute retry delay
                        continue
                        
                # Normal interval wait
                await asyncio.sleep(self.interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("Refresh loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous refresh: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
        self.is_running = False
        logger.info("Automated refresh stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current refresh status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {
            "is_running": False,
            "current_job": None,
            "next_run": None,
            "interval_seconds": self.interval_seconds
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "success_rate": 0,
            "average_duration": 0,
            "average_rows_processed": 0,
            "current_row_count": 0,
            "last_success": None,
            "last_failure": None
        }

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Automated Refresh System')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Refresh interval in seconds (default: 3600)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum consecutive retries on failure (default: 3)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    
    args = parser.parse_args()
    
    manager = AutomatedRefreshManager(
        interval_seconds=args.interval,
        max_retries=args.max_retries
    )
    
    if args.once:
        # Run single refresh
        job = await manager.run_refresh()
        print(f"Refresh completed with status: {job.status.value}")
        if job.error_message:
            print(f"Error: {job.error_message}")
    else:
        # Run continuous
        await manager.run_continuous()

if __name__ == "__main__":
    asyncio.run(main())
