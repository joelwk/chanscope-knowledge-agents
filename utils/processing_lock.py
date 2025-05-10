"""
Process Lock Utility for Replit and Docker environments

This module provides utilities for managing process locks across Replit
development and deployment environments using Replit Object Storage,
with fallback to file-based locks for Docker/local environments.
"""

import os
import json
import time
import fcntl
import errno
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Lock key constants
PROCESSING_LOCK_KEY = "chanscope_data_processing_lock"
INIT_COMPLETE_MARKER_KEY = "chanscope_init_complete_marker"
LOCK_EXPIRY_TIME = 600  # 10 minutes in seconds
INIT_MARKER_VALID_TIME = 3600  # 1 hour in seconds

# File-based lock constants
FILE_LOCK_DIR = Path("data")
FILE_LOCK_PATH = FILE_LOCK_DIR / ".process_lock"
FILE_INIT_MARKER_PATH = FILE_LOCK_DIR / ".init_complete_marker"


class ProcessLockManager:
    """
    Manages process locks using Replit Object Storage with fallback to file-based locks.
    
    This class provides a unified interface for process locking that works across
    different environments:
    - In Replit: Uses Object Storage for persistence across restarts
    - In Docker/local: Uses file-based locks with fcntl
    """
    
    def __init__(self):
        """Initialize the process lock manager."""
        self.client = None
        self.has_object_storage = False
        
        # First check if we're explicitly in Docker - prioritize this check
        is_docker = os.environ.get('DOCKER_ENV', '').lower() in ('true', '1', 'yes')
        
        # Then check for Replit environment
        is_replit = os.environ.get('REPLIT_ENV', '').lower() in ('replit', 'true', '1', 'yes') or os.environ.get('REPL_ID') is not None
        
        # Log the detected environment
        if is_docker:
            logger.info("Docker environment detected, using file-based locks")
        elif is_replit:
            logger.info("Replit environment detected, attempting to use Object Storage")
        else:
            logger.info("Local environment detected, using file-based locks")
        
        # Only try Object Storage in Replit environment and not Docker
        if is_replit and not is_docker:
            try:
                # Try to import and initialize Object Storage client
                from replit.object_storage import Client
                self.client = Client()
                self.has_object_storage = True
                logger.info("Successfully initialized Replit Object Storage client")
            except ImportError:
                logger.info("Replit Object Storage package not available - using file-based locks")
            except Exception as e:
                logger.error(f"Error initializing Replit Object Storage: {e}")
                logger.info("Falling back to file-based locks")
        
        # Ensure lock directory exists for file-based locks
        FILE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        
        # Log which lock method we're using
        if self.has_object_storage:
            logger.info("Using Replit Object Storage for process locks")
        else:
            logger.info("Using file-based locks for process locking")
    
    def acquire_lock(self) -> bool:
        """
        Attempt to acquire a processing lock.
        
        Returns:
            bool: True if lock was acquired, False otherwise
        """
        if self.has_object_storage and self.client:
            return self._acquire_object_storage_lock()
        else:
            return self._acquire_file_lock()
    
    def _acquire_object_storage_lock(self) -> bool:
        """Acquire lock using Replit Object Storage."""
        try:
            # Check if lock exists and is recent
            if self.client.exists(PROCESSING_LOCK_KEY):
                try:
                    lock_info = json.loads(self.client.download_as_text(PROCESSING_LOCK_KEY))
                    timestamp = lock_info.get("timestamp")
                    pid = lock_info.get("pid")
                    start_time = lock_info.get("start_time", "unknown")
                    
                    # If lock is recent (less than expiry time), exit
                    if timestamp and (time.time() - timestamp < LOCK_EXPIRY_TIME):
                        logger.info(f"Data processing already running (started at {start_time}, PID: {pid})")
                        return False
                    else:
                        logger.info(f"Found expired lock from PID {pid}, will override")
                except Exception as e:
                    logger.error(f"Error reading lock: {e}")
            
            # Set lock with current timestamp
            lock_data = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "start_time": datetime.now().isoformat()
            }
            self.client.upload_from_text(PROCESSING_LOCK_KEY, json.dumps(lock_data))
            logger.info(f"Acquired processing lock in Object Storage (PID: {os.getpid()})")
            return True
            
        except Exception as e:
            logger.error(f"Error acquiring Object Storage lock: {e}")
            logger.info("Falling back to file-based lock")
            return self._acquire_file_lock()  # Fall back to file lock if Object Storage fails
    
    def _acquire_file_lock(self) -> bool:
        """Acquire lock using file-based locking."""
        try:
            # Check if the lock file exists and contains recent data
            if FILE_LOCK_PATH.exists():
                try:
                    with open(FILE_LOCK_PATH, 'r') as f:
                        lock_info = json.loads(f.read())
                        timestamp = lock_info.get("timestamp")
                        pid = lock_info.get("pid")
                        start_time = lock_info.get("start_time", "unknown")
                        
                        # Check if the process is still running
                        if pid and self._is_process_running(pid):
                            # If lock is recent (less than expiry time), exit
                            if timestamp and (time.time() - timestamp < LOCK_EXPIRY_TIME):
                                logger.info(f"Data processing already running (started at {start_time}, PID: {pid})")
                                return False
                        
                        # If the process is not running or lock is expired, we can override
                        logger.info(f"Found expired or stale lock from PID {pid}, will override")
                        
                        # Try to clean up the stale lock file
                        try:
                            os.remove(FILE_LOCK_PATH)
                            logger.info("Removed stale lock file")
                        except Exception as e:
                            logger.warning(f"Could not remove stale lock file: {e}")
                            
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.warning(f"Lock file exists but couldn't be read: {e}")
                    # Try to remove invalid lock file
                    try:
                        os.remove(FILE_LOCK_PATH)
                        logger.info("Removed invalid lock file")
                    except Exception as e2:
                        logger.warning(f"Could not remove invalid lock file: {e2}")
                except Exception as e:
                    logger.error(f"Error checking lock file: {e}")
            
            # Create lock file with advisory locking
            lock_fd = None
            try:
                # Ensure the lock directory exists
                FILE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
                
                # Create or open the lock file - use 'w+' to truncate and allow reading
                lock_fd = open(FILE_LOCK_PATH, 'w+')
                
                # Try to acquire a non-blocking exclusive lock
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (AttributeError, OSError) as e:
                    # Handle Windows systems that don't have fcntl
                    if os.name == 'nt':
                        # Write PID to indicate lock on Windows
                        lock_data = {
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "start_time": datetime.now().isoformat()
                        }
                        lock_fd.write(json.dumps(lock_data))
                        lock_fd.flush()
                        self.lock_fd = lock_fd
                        logger.info(f"Acquired file-based processing lock on Windows (PID: {os.getpid()})")
                        return True
                    else:
                        # Re-raise the exception for non-Windows systems
                        raise
                
                # Write lock information
                lock_data = {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "start_time": datetime.now().isoformat()
                }
                lock_fd.write(json.dumps(lock_data))
                lock_fd.flush()
                
                # Keep the file descriptor open to maintain the lock
                # Store it as an attribute for later release
                self.lock_fd = lock_fd
                
                logger.info(f"Acquired file-based processing lock (PID: {os.getpid()})")
                return True
                
            except IOError as e:
                # Failed to acquire lock
                if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
                    logger.info("Another process already holds the lock")
                    if lock_fd:
                        lock_fd.close()
                    return False
                else:
                    logger.error(f"Error acquiring file lock: {e}")
                    if lock_fd:
                        lock_fd.close()
                    return False  # Changed from True to False to prevent processing if lock acquisition fails
            
        except Exception as e:
            logger.error(f"Unexpected error in file-based lock acquisition: {e}")
            return False  # Changed from True to False for more predictable behavior

    def _is_process_running(self, pid):
        """Check if a process with the given PID is running."""
        try:
            # Different approach for Windows vs Unix-like systems
            if os.name == 'nt':  # Windows
                import ctypes
                kernel32 = ctypes.windll.kernel32
                process = kernel32.OpenProcess(1, 0, pid)
                if process != 0:
                    kernel32.CloseHandle(process)
                    return True
                return False
            else:  # Unix-like
                # No signal is sent, just check if the process exists
                os.kill(pid, 0)
                return True
        except (OSError, AttributeError):
            return False
    
    def release_lock(self) -> bool:
        """
        Release the processing lock.
        
        Returns:
            bool: True if lock was released, False if error occurred
        """
        if self.has_object_storage and self.client:
            result = self._release_object_storage_lock()
            if not result:
                # Try file lock as fallback
                return self._release_file_lock()
            return result
        else:
            return self._release_file_lock()
    
    def _release_object_storage_lock(self) -> bool:
        """Release lock from Replit Object Storage."""
        try:
            if self.client.exists(PROCESSING_LOCK_KEY):
                self.client.delete(PROCESSING_LOCK_KEY)
                logger.info("Released processing lock from Object Storage")
            return True
        except Exception as e:
            logger.error(f"Error releasing Object Storage lock: {e}")
            return False
    
    def _release_file_lock(self) -> bool:
        """Release file-based lock."""
        try:
            # Release the lock by closing the file descriptor
            if hasattr(self, 'lock_fd') and self.lock_fd:
                try:
                    # Only use flock on non-Windows systems
                    if os.name != 'nt':
                        try:
                            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                        except (AttributeError, OSError) as e:
                            logger.warning(f"Could not use flock to unlock: {e}")
                    
                    # Always close the file descriptor
                    self.lock_fd.close()
                    
                    # We don't delete the file so it can store the last successful process info
                    logger.info("Released file-based processing lock")
                    delattr(self, 'lock_fd')
                    return True
                except Exception as e:
                    logger.error(f"Error releasing file lock: {e}")
                    return False
            else:
                logger.warning("No file lock to release")
                return True
        except Exception as e:
            logger.error(f"Error in file lock release: {e}")
            return False
    
    def mark_initialization_complete(self, status: bool = True, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark the initialization as complete.
        
        Args:
            status: Whether initialization was successful
            metadata: Additional metadata to store
            
        Returns:
            bool: True if marker was set, False if error occurred
        """
        if self.has_object_storage and self.client:
            result = self._mark_object_storage_initialization(status, metadata)
            if not result:
                # Try file-based as fallback
                return self._mark_file_initialization(status, metadata)
            return result
        else:
            return self._mark_file_initialization(status, metadata)
    
    def _mark_object_storage_initialization(self, status: bool, metadata: Optional[Dict[str, Any]]) -> bool:
        """Mark initialization complete in Object Storage."""
        try:
            marker_data = {
                "timestamp": time.time(),
                "status": "success" if status else "error",
                "data_ready": status,
                "completion_time": datetime.now().isoformat()
            }
            
            # Add any additional metadata
            if metadata:
                marker_data.update(metadata)
                
            self.client.upload_from_text(INIT_COMPLETE_MARKER_KEY, json.dumps(marker_data))
            logger.info(f"Marked initialization as {'complete' if status else 'failed'} in Object Storage")
            return True
        except Exception as e:
            logger.error(f"Error setting initialization marker in Object Storage: {e}")
            return False
    
    def _mark_file_initialization(self, status: bool, metadata: Optional[Dict[str, Any]]) -> bool:
        """Mark initialization complete using file-based marker."""
        try:
            marker_data = {
                "timestamp": time.time(),
                "status": "success" if status else "error",
                "data_ready": status,
                "completion_time": datetime.now().isoformat()
            }
            
            # Add any additional metadata
            if metadata:
                marker_data.update(metadata)
            
            # Write to the marker file
            with open(FILE_INIT_MARKER_PATH, 'w') as f:
                f.write(json.dumps(marker_data))
            
            logger.info(f"Marked initialization as {'complete' if status else 'failed'} in file system")
            return True
        except Exception as e:
            logger.error(f"Error setting file-based initialization marker: {e}")
            return False
    
    def check_initialization_status(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if initialization was completed recently.
        
        Returns:
            Tuple of (initialization_needed, marker_data)
            - initialization_needed: True if initialization is needed, False otherwise
            - marker_data: The marker data if it exists, None otherwise
        """
        if self.has_object_storage and self.client:
            try:
                result = self._check_object_storage_initialization()
                return result
            except Exception as e:
                logger.error(f"Error checking Object Storage initialization, falling back to file: {e}")
                return self._check_file_initialization()
        else:
            return self._check_file_initialization()
    
    def _check_object_storage_initialization(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check initialization status in Object Storage."""
        try:
            if self.client.exists(INIT_COMPLETE_MARKER_KEY):
                marker_data = json.loads(self.client.download_as_text(INIT_COMPLETE_MARKER_KEY))
                timestamp = marker_data.get("timestamp", 0)
                status = marker_data.get("status", "unknown")
                completion_time = marker_data.get("completion_time", "unknown")
                
                # If marker is recent (less than valid time) and successful, initialization not needed
                if timestamp and (time.time() - timestamp < INIT_MARKER_VALID_TIME) and status == "success":
                    logger.info(f"Initialization completed recently ({completion_time}), skipping")
                    return False, marker_data
                else:
                    if status != "success":
                        logger.info(f"Previous initialization failed with status {status}, will retry")
                    else:
                        logger.info(f"Previous initialization is outdated (completed at {completion_time})")
                    return True, marker_data
            else:
                logger.info("No initialization marker found in Object Storage, initialization needed")
                return True, None
        except Exception as e:
            logger.error(f"Error checking initialization status in Object Storage: {e}")
            return True, None
    
    def _check_file_initialization(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check initialization status using file-based marker."""
        try:
            if FILE_INIT_MARKER_PATH.exists():
                try:
                    with open(FILE_INIT_MARKER_PATH, 'r') as f:
                        marker_data = json.loads(f.read())
                        
                    timestamp = marker_data.get("timestamp", 0)
                    status = marker_data.get("status", "unknown")
                    completion_time = marker_data.get("completion_time", "unknown")
                    
                    # If marker is recent (less than valid time) and successful, initialization not needed
                    if timestamp and (time.time() - timestamp < INIT_MARKER_VALID_TIME) and status == "success":
                        logger.info(f"Initialization completed recently ({completion_time}), skipping")
                        return False, marker_data
                    else:
                        if status != "success":
                            logger.info(f"Previous initialization failed with status {status}, will retry")
                        else:
                            logger.info(f"Previous initialization is outdated (completed at {completion_time})")
                        return True, marker_data
                        
                except json.JSONDecodeError:
                    logger.warning("Initialization marker file exists but couldn't be parsed")
                    return True, None
            else:
                logger.info("No file-based initialization marker found, initialization needed")
                return True, None
        except Exception as e:
            logger.error(f"Error checking file-based initialization status: {e}")
            return True, None
    
    def force_cleanup(self) -> bool:
        """
        Force cleanup of any locks, regardless of owner.
        Use with caution - only call this when you're sure no other process is using the lock.
        
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        success = True
        
        # First try to release our own lock if we have one
        if hasattr(self, 'lock_fd') and self.lock_fd:
            try:
                self.release_lock()
            except Exception as e:
                logger.error(f"Error releasing our own lock: {e}")
                success = False
        
        # Clean up Object Storage locks if available
        if self.has_object_storage and self.client:
            try:
                if self.client.exists(PROCESSING_LOCK_KEY):
                    self.client.delete(PROCESSING_LOCK_KEY)
                    logger.info("Cleaned up Object Storage lock")
            except Exception as e:
                logger.error(f"Error cleaning up Object Storage lock: {e}")
                success = False
        
        # Clean up file-based locks
        try:
            if FILE_LOCK_PATH.exists():
                try:
                    # Try to read the lock file to log what we're cleaning up
                    with open(FILE_LOCK_PATH, 'r') as f:
                        try:
                            lock_info = json.loads(f.read())
                            pid = lock_info.get("pid", "unknown")
                            start_time = lock_info.get("start_time", "unknown")
                            logger.info(f"Cleaning up lock from PID {pid} (started at {start_time})")
                        except:
                            logger.info("Cleaning up unreadable lock file")
                    
                    # Make sure no file descriptors are open to this file from this process
                    if hasattr(self, 'lock_fd') and self.lock_fd:
                        try:
                            self.lock_fd.close()
                            delattr(self, 'lock_fd')
                        except:
                            pass
                    
                    # Remove the lock file
                    # Use os.remove instead of Path.unlink() for better compatibility
                    os.remove(FILE_LOCK_PATH)
                    logger.info("Removed lock file")
                    
                except Exception as e:
                    logger.error(f"Error removing lock file: {e}")
                    success = False
        except Exception as e:
            logger.error(f"Error in file lock cleanup: {e}")
            success = False
            
        return success 