#!/usr/bin/env python
"""
Unified Control Script for Automated Refresh System
Provides command-line control for the refresh system
"""

import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:5000/api"

def get_status():
    """Get current system status"""
    try:
        response = requests.get(f"{API_URL}/status")
        status = response.json()
        
        print("\n=== ChanScope Refresh System Status ===")
        print(f"Status: {'Running' if status['is_running'] else 'Idle'}")
        print(f"Current Job: {status.get('current_job', {}).get('id', 'None')}")
        print(f"Next Run: {status.get('next_run', 'Not scheduled')}")
        print(f"Interval: {status.get('interval_seconds', 3600) / 60:.0f} minutes")
        
        # Get metrics
        response = requests.get(f"{API_URL}/metrics")
        metrics = response.json()
        
        print("\n=== Performance Metrics ===")
        print(f"Total Runs: {metrics.get('total_runs', 0)}")
        print(f"Successful: {metrics.get('successful_runs', 0)}")
        print(f"Failed: {metrics.get('failed_runs', 0)}")
        print(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"Avg Duration: {metrics.get('average_duration', 0):.1f} seconds")
        print(f"Avg Rows Processed: {metrics.get('average_rows_processed', 0):.0f}")
        
        if metrics.get('last_success'):
            print(f"Last Success: {metrics['last_success']}")
        if metrics.get('last_failure'):
            print(f"Last Failure: {metrics['last_failure']}")
            
    except Exception as e:
        print(f"Error getting status: {e}")
        print("Is the dashboard server running?")
        return False
    
    return True

def start_refresh(interval=3600):
    """Start automatic refresh"""
    try:
        # Update interval if provided
        if interval != 3600:
            response = requests.post(f"{API_URL}/config", 
                                    json={"interval_seconds": interval, "max_retries": 3})
            if response.status_code == 200:
                print(f"Updated interval to {interval} seconds")
        
        # Start refresh
        response = requests.post(f"{API_URL}/control", json={"action": "start"})
        result = response.json()
        
        if result['status'] == 'started':
            print("Auto-refresh started successfully")
        elif result['status'] == 'already_running':
            print("Auto-refresh is already running")
        else:
            print(f"Unexpected status: {result['status']}")
            
    except Exception as e:
        print(f"Error starting refresh: {e}")
        print("Is the dashboard server running?")
        return False
    
    return True

def stop_refresh():
    """Stop automatic refresh"""
    try:
        response = requests.post(f"{API_URL}/control", json={"action": "stop"})
        result = response.json()
        
        if result['status'] == 'stopped':
            print("Auto-refresh stopped successfully")
        elif result['status'] == 'not_running':
            print("Auto-refresh was not running")
        else:
            print(f"Unexpected status: {result['status']}")
            
    except Exception as e:
        print(f"Error stopping refresh: {e}")
        print("Is the dashboard server running?")
        return False
    
    return True

def run_once():
    """Run a single refresh"""
    try:
        response = requests.post(f"{API_URL}/control", json={"action": "refresh_once"})
        result = response.json()
        
        if result['status'] == 'refresh_triggered':
            print("Manual refresh triggered successfully")
            print("Check status for progress...")
        else:
            print(f"Unexpected status: {result['status']}")
            
    except Exception as e:
        print(f"Error running refresh: {e}")
        print("Is the dashboard server running?")
        return False
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Control the ChanScope Automated Refresh System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/refresh_control.py status          # Check current status
  python scripts/refresh_control.py start           # Start auto-refresh with default interval
  python scripts/refresh_control.py start --interval 1800  # Start with 30-minute interval
  python scripts/refresh_control.py stop            # Stop auto-refresh
  python scripts/refresh_control.py run-once        # Run a single refresh
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Status command
    subparsers.add_parser('status', help='Get current status and metrics')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start automatic refresh')
    start_parser.add_argument('--interval', type=int, default=3600,
                            help='Refresh interval in seconds (default: 3600)')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop automatic refresh')
    
    # Run once command
    subparsers.add_parser('run-once', help='Run a single refresh')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'status':
        get_status()
    elif args.command == 'start':
        start_refresh(args.interval)
    elif args.command == 'stop':
        stop_refresh()
    elif args.command == 'run-once':
        run_once()

if __name__ == "__main__":
    main()