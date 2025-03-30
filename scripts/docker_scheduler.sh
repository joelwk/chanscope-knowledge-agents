#!/bin/bash
# Docker Data Scheduler Manager for Chanscope
# This script handles data scheduler management in Docker environment

set -e

# Configuration
APP_ROOT="${APP_ROOT:-/app}"
DATA_DIR="${DATA_DIR:-${APP_ROOT}/data}"
LOGS_DIR="${LOGS_DIR:-${APP_ROOT}/logs}"
SCHEDULER_LOG="${LOGS_DIR}/scheduler.log"
PID_FILE="${DATA_DIR}/.scheduler_pid"

# Default settings
DEFAULT_UPDATE_INTERVAL=3600  # 1 hour in seconds

# Ensure log directory exists
mkdir -p "${LOGS_DIR}"

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if scheduler is running
is_scheduler_running() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ps -p "${PID}" > /dev/null; then
            return 0  # Running
        else
            return 1  # Not running but PID file exists
        fi
    else
        return 2  # PID file doesn't exist
    fi
}

# Function to start the scheduler
start_scheduler() {
    local interval=${1:-$DEFAULT_UPDATE_INTERVAL}
    
    if is_scheduler_running; then
        echo -e "${YELLOW}Scheduler is already running with PID: $(cat ${PID_FILE})${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Starting data scheduler with update interval: ${interval} seconds${NC}"
    
    # Create the scheduler background process
    cd "${APP_ROOT}" && python scripts/scheduled_update.py refresh --continuous --interval=$interval > "${SCHEDULER_LOG}" 2>&1 &
    
    SCHEDULER_PID=$!
    echo ${SCHEDULER_PID} > "${PID_FILE}"
    
    echo -e "${GREEN}Data scheduler started with PID: ${SCHEDULER_PID}${NC}"
    echo -e "${GREEN}Update interval: ${interval} seconds${NC}"
    echo -e "${YELLOW}Logs: ${SCHEDULER_LOG}${NC}"
    return 0
}

# Function to stop the scheduler
stop_scheduler() {
    if ! is_scheduler_running; then
        echo -e "${YELLOW}Scheduler is not running.${NC}"
        [ -f "${PID_FILE}" ] && rm "${PID_FILE}"
        return 0
    fi
    
    PID=$(cat "${PID_FILE}")
    echo -e "${YELLOW}Stopping scheduler with PID: ${PID}${NC}"
    
    # Kill the scheduler process and its children
    pkill -P ${PID} 2>/dev/null || true
    kill ${PID} 2>/dev/null || true
    
    # Wait for the process to terminate
    for i in {1..5}; do
        if ! ps -p ${PID} > /dev/null; then
            break
        fi
        echo "Waiting for scheduler to terminate... (${i}/5)"
        sleep 1
    done
    
    # Force kill if still running
    if ps -p ${PID} > /dev/null; then
        echo -e "${YELLOW}Force killing scheduler process...${NC}"
        kill -9 ${PID} 2>/dev/null || true
    fi
    
    rm -f "${PID_FILE}"
    echo -e "${GREEN}Scheduler stopped.${NC}"
    return 0
}

# Function to show scheduler status
show_status() {
    if is_scheduler_running; then
        PID=$(cat "${PID_FILE}")
        echo -e "${GREEN}Scheduler is running with PID: ${PID}${NC}"
        
        # Show process info
        ps -f -p ${PID} 2>/dev/null || echo "Process details not available"
        
        # Show the last few log entries
        echo -e "\nLast 10 log entries:"
        tail -n 10 "${SCHEDULER_LOG}" 2>/dev/null || echo "No logs found."
        
        return 0
    elif [ -f "${PID_FILE}" ]; then
        echo -e "${RED}Scheduler is not running but PID file exists.${NC}"
        echo "Clean up with: $0 cleanup"
        return 1
    else
        echo -e "${YELLOW}Scheduler is not running.${NC}"
        return 2
    fi
}

# Function to clean up stale PID files
cleanup() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ! ps -p ${PID} > /dev/null; then
            echo -e "${YELLOW}Removing stale PID file...${NC}"
            rm -f "${PID_FILE}"
        else
            echo -e "${YELLOW}PID file points to a running process. Stop the scheduler first.${NC}"
            return 1
        fi
    fi
    echo -e "${GREEN}Cleanup completed.${NC}"
    return 0
}

# Function to run a single update manually
run_update() {
    echo -e "${YELLOW}Running manual data update...${NC}"
    cd "${APP_ROOT}" && python scripts/scheduled_update.py refresh
    return $?
}

# Display help
show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start [interval]    Start the scheduler (interval in seconds, default: ${DEFAULT_UPDATE_INTERVAL})"
    echo "  stop                Stop the scheduler"
    echo "  restart [interval]  Restart the scheduler"
    echo "  status              Show scheduler status"
    echo "  logs [lines]        Show recent logs (default: 50 lines)"
    echo "  update              Run a data update manually"
    echo "  cleanup             Clean up stale PID files"
    echo "  help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start with default interval (1 hour)"
    echo "  $0 start 1800      # Start with 30-minute interval"
    echo "  $0 logs 100        # Show last 100 log lines"
    echo ""
}

# Function to show recent logs
show_logs() {
    local lines=${1:-50}
    if [ -f "${SCHEDULER_LOG}" ]; then
        echo -e "${YELLOW}Showing last ${lines} lines of scheduler logs:${NC}"
        tail -n ${lines} "${SCHEDULER_LOG}"
    else
        echo -e "${RED}Scheduler log file not found: ${SCHEDULER_LOG}${NC}"
        return 1
    fi
    return 0
}

# Main command handling
case "$1" in
    start)
        start_scheduler ${2:-$DEFAULT_UPDATE_INTERVAL}
        ;;
    stop)
        stop_scheduler
        ;;
    restart)
        stop_scheduler
        sleep 1
        start_scheduler ${2:-$DEFAULT_UPDATE_INTERVAL}
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs ${2:-50}
        ;;
    update)
        run_update
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac

exit $? 