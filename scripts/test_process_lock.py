#!/usr/bin/env python3
"""
Test script for ProcessLockManager utility

This script tests the ProcessLockManager utility to ensure that
it correctly handles process locking and status tracking.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
import platform
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Detect environment properly before importing ProcessLockManager
def detect_environment():
    """
    Detect the current execution environment.
    Returns a tuple of (is_docker, is_replit)
    """
    is_docker = False
    is_replit = False
    
    # Check for Docker environment
    if os.environ.get('DOCKER_ENV', '').lower() in ('true', '1', 'yes'):
        is_docker = True
    elif platform.system() != 'Windows':  # Only check these on non-Windows systems
        # Check for .dockerenv file
        if Path('/.dockerenv').exists():
            is_docker = True
        # Check for docker in cgroup
        try:
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    is_docker = True
        except (FileNotFoundError, PermissionError):
            pass
    
    # Check for Replit environment
    if os.environ.get('REPLIT_ENV', '').lower() in ('replit', 'true', '1', 'yes') or os.environ.get('REPL_ID') is not None:
        is_replit = True
    
    return is_docker, is_replit

# Set environment variables explicitly
is_docker, is_replit = detect_environment()

if is_docker:
    print("Running in Docker environment")
    os.environ['DOCKER_ENV'] = 'true'
    if 'REPLIT_ENV' in os.environ:
        del os.environ['REPLIT_ENV']
    if 'REPL_ID' in os.environ:
        del os.environ['REPL_ID']
elif is_replit:
    print("Running in Replit environment")
    os.environ['REPLIT_ENV'] = 'replit'
    if 'DOCKER_ENV' in os.environ:
        del os.environ['DOCKER_ENV']
else:
    print(f"Running in local environment (OS: {platform.system()})")
    # Clear any environment variables that might cause confusion
    if 'REPLIT_ENV' in os.environ:
        del os.environ['REPLIT_ENV']
    if 'REPL_ID' in os.environ:
        del os.environ['REPL_ID']
    if 'DOCKER_ENV' in os.environ:
        del os.environ['DOCKER_ENV']

# Import ProcessLockManager after environment setup
from scripts.utils.processing_lock import ProcessLockManager, FILE_INIT_MARKER_PATH, FILE_LOCK_PATH

def test_lock_acquisition(lock_name=None):
    """Test lock acquisition and release."""
    print(f"Testing lock acquisition at {datetime.now().isoformat()}")
    
    # Create lock manager
    lock_manager = ProcessLockManager()
    
    # Try to acquire the lock
    acquired = lock_manager.acquire_lock()
    if acquired:
        print(f"✅ Successfully acquired lock (PID: {os.getpid()})")
        
        # Hold the lock for a while
        print("Holding lock for 10 seconds...")
        time.sleep(10)
        
        # Release the lock
        released = lock_manager.release_lock()
        if released:
            print("✅ Successfully released lock")
            return True
        else:
            print("❌ Failed to release lock")
            return False
    else:
        print("❌ Failed to acquire lock - another process is running")
        return False
    
    return acquired

def test_lock_contention():
    """Test lock contention with a child process."""
    print(f"Testing lock contention at {datetime.now().isoformat()}")
    
    # Create lock manager
    lock_manager = ProcessLockManager()
    
    # Clean up any existing locks first
    print("Cleaning up any existing locks...")
    lock_manager.force_cleanup()
    
    # Wait a moment to ensure cleanup is complete
    time.sleep(1)
    
    # Try to acquire the lock in the parent process first
    acquired = lock_manager.acquire_lock()
    if not acquired:
        print("❌ Failed to acquire lock in parent process despite cleanup")
        return False
    
    print(f"✅ Acquired lock in parent process (PID: {os.getpid()})")
    
    # Create a child process that tries to acquire the lock
    try:
        # Pass the current environment to the child process
        env = os.environ.copy()
        
        # Set up child process
        if platform.system() == 'Windows':
            python_exe = sys.executable
            script_path = os.path.abspath(__file__)
            args = [python_exe, script_path, "--test-acquire"]
            
            # On Windows, we need to use a different creation flags approach
            child_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # For Linux/Docker, use a simpler approach
            child_process = subprocess.Popen(
                [sys.executable, __file__, "--test-acquire"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
        
        # Wait for the child process to finish (with timeout)
        try:
            stdout, stderr = child_process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            print("Child process timed out, killing it")
            child_process.kill()
            stdout, stderr = child_process.communicate()
        
        # Print the child process output
        print("\nChild process output:")
        print(stdout)
        
        # Print stderr if there was any output
        if stderr.strip():
            print("\nChild process errors:")
            print(stderr)
        
        # Check if the child process failed to acquire the lock (expected behavior)
        # The process should exit with non-zero status or explicitly report lock failure
        lock_acquired = "Successfully acquired lock" in stdout
        child_exit_code = child_process.returncode
        
        if lock_acquired:
            print("❌ Child process was able to acquire the lock (should have been denied)")
            success = False
        else:
            print(f"✅ Child process correctly denied the lock (exit code: {child_exit_code})")
            success = True
        
        # Release the lock in the parent process
        released = lock_manager.release_lock()
        if released:
            print("✅ Released lock in parent process")
        else:
            print("❌ Failed to release lock in parent process")
            success = False
        
        return success
    
    except Exception as e:
        print(f"❌ Error in lock contention test: {e}")
        # Try to release the lock before exiting
        lock_manager.release_lock()
        return False

def test_initialization_marker():
    """Test setting and checking initialization markers."""
    print(f"Testing initialization markers at {datetime.now().isoformat()}")
    
    # Create lock manager
    lock_manager = ProcessLockManager()
    
    # Set initialization marker
    marker_set = lock_manager.mark_initialization_complete(
        status=True,
        metadata={
            "test_key": "test_value",
            "pid": os.getpid(),
            "test_time": datetime.now().isoformat()
        }
    )
    
    if marker_set:
        print("✅ Successfully set initialization marker")
    else:
        print("❌ Failed to set initialization marker")
        return False
    
    # Check initialization status
    needs_init, marker_data = lock_manager.check_initialization_status()
    
    if marker_data:
        print(f"✅ Successfully retrieved marker data: {marker_data}")
    else:
        print("❌ Failed to retrieve marker data")
        return False
    
    if not needs_init:
        print("✅ Correctly reports initialization is not needed")
    else:
        print("❌ Incorrectly reports initialization is needed")
        return False
    
    # Test with failed status
    marker_set = lock_manager.mark_initialization_complete(
        status=False,
        metadata={
            "error": "Test error message",
            "pid": os.getpid(),
            "test_time": datetime.now().isoformat()
        }
    )
    
    if marker_set:
        print("✅ Successfully set failed initialization marker")
    else:
        print("❌ Failed to set failed initialization marker")
        return False
    
    # Check initialization status again
    needs_init, marker_data = lock_manager.check_initialization_status()
    
    if needs_init:
        print("✅ Correctly reports initialization is needed after failure")
    else:
        print("❌ Incorrectly reports initialization is not needed after failure")
        return False
    
    return True

def setup_test_environment():
    """Prepare the environment for testing."""
    print("Setting up test environment...")
    
    # Create a lock manager for cleanup
    lock_manager = ProcessLockManager()
    
    # Clean up any existing locks
    lock_manager.force_cleanup()
    
    # Clean up any initialization markers
    try:
        if os.path.exists(FILE_INIT_MARKER_PATH):
            os.remove(FILE_INIT_MARKER_PATH)
            print("Removed initialization marker file")
    except Exception as e:
        print(f"Warning: Could not remove initialization marker: {e}")
    
    # Wait a moment to ensure cleanup is complete
    time.sleep(1)
    
    print("Test environment ready")
    return True

def run_all_tests():
    """Run all tests."""
    print("Running all ProcessLockManager tests...")
    
    # Setup the test environment
    setup_success = setup_test_environment()
    if not setup_success:
        print("❌ Failed to set up test environment")
        return False
    
    # Test lock acquisition
    print("\n1. Testing basic lock acquisition and release...")
    acquisition_success = test_lock_acquisition()
    
    # Clean up between tests
    lock_manager = ProcessLockManager()
    lock_manager.force_cleanup()
    print("Cleaned up after lock acquisition test")
    
    # Sleep briefly to ensure any locks are fully released
    time.sleep(2)
    
    # Test lock contention
    print("\n2. Testing lock contention with child process...")
    contention_success = test_lock_contention()
    
    # Clean up between tests
    lock_manager = ProcessLockManager()
    lock_manager.force_cleanup()
    print("Cleaned up after lock contention test")
    
    # Sleep briefly to ensure any locks are fully released
    time.sleep(2)
    
    # Test initialization marker
    print("\n3. Testing initialization markers...")
    marker_success = test_initialization_marker()
    
    # Clean up after all tests
    lock_manager = ProcessLockManager()
    lock_manager.force_cleanup()
    print("Cleaned up after all tests")
    
    # Report results
    print("\nTest Results:")
    print(f"- Lock acquisition: {'✅ PASS' if acquisition_success else '❌ FAIL'}")
    print(f"- Lock contention: {'✅ PASS' if contention_success else '❌ FAIL'}")
    print(f"- Initialization markers: {'✅ PASS' if marker_success else '❌ FAIL'}")
    
    all_passed = acquisition_success and contention_success and marker_success
    print(f"\nOverall result: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    return all_passed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ProcessLockManager utility")
    
    parser.add_argument(
        "--test-acquire",
        action="store_true",
        help="Test lock acquisition (for child process)"
    )
    
    parser.add_argument(
        "--test-marker",
        action="store_true",
        help="Test initialization marker"
    )
    
    parser.add_argument(
        "--test-contention",
        action="store_true",
        help="Test lock contention"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Run the requested tests
    success = False
    if args.test_acquire:
        success = test_lock_acquisition()
    elif args.test_marker:
        success = test_initialization_marker()
    elif args.test_contention:
        success = test_lock_contention()
    elif args.all:
        success = run_all_tests()
    else:
        # Default to running all tests
        success = run_all_tests()
    
    # Set exit code based on test success/failure
    sys.exit(0 if success else 1) 