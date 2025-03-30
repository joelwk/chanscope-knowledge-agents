#!/bin/bash
set -e

# Get the application root directory
APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${APP_ROOT}/data"
LOGS_DIR="${DATA_DIR}/logs"
SCHEDULER_LOG="${LOGS_DIR}/scheduler.log"
PID_FILE="${DATA_DIR}/.scheduler_pid"

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Default settings
DEFAULT_UPDATE_INTERVAL=3600  # 1 hour in seconds

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
        echo "Scheduler is already running with PID: $(cat ${PID_FILE})"
        return 0
    fi
    
    # Create the scheduler script with the specified interval
    cat > "${APP_ROOT}/scripts/run_scheduler.sh" << EOF
#!/bin/bash
echo "Starting data scheduler with update interval: ${interval} seconds"
while true; do
    echo "[$(date)] Running scheduled data update..." >> "${SCHEDULER_LOG}" 2>&1
    cd "${APP_ROOT}" && poetry run python scripts/scheduled_update.py refresh >> "${SCHEDULER_LOG}" 2>&1
    echo "[$(date)] Scheduled update completed, sleeping for ${interval} seconds" >> "${SCHEDULER_LOG}" 2>&1
    sleep ${interval}
done
EOF
    
    chmod +x "${APP_ROOT}/scripts/run_scheduler.sh"
    
    # Start the scheduler in the background
    nohup "${APP_ROOT}/scripts/run_scheduler.sh" >> "${SCHEDULER_LOG}" 2>&1 &
    SCHEDULER_PID=$!
    echo ${SCHEDULER_PID} > "${PID_FILE}"
    
    echo "Data scheduler started with PID: ${SCHEDULER_PID}"
    echo "Update interval: ${interval} seconds"
    echo "Logs: ${SCHEDULER_LOG}"
    return 0
}

# Function to stop the scheduler
stop_scheduler() {
    if ! is_scheduler_running; then
        echo "Scheduler is not running."
        [ -f "${PID_FILE}" ] && rm "${PID_FILE}"
        return 0
    fi
    
    PID=$(cat "${PID_FILE}")
    echo "Stopping scheduler with PID: ${PID}"
    
    # Kill the scheduler process and its children
    pkill -P ${PID} || true
    kill ${PID} || true
    
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
        echo "Force killing scheduler process..."
        kill -9 ${PID} || true
    fi
    
    rm -f "${PID_FILE}"
    echo "Scheduler stopped."
    return 0
}

# Function to show scheduler status
show_status() {
    if is_scheduler_running; then
        PID=$(cat "${PID_FILE}")
        echo "Scheduler is running with PID: ${PID}"
        
        # Show process info
        ps -f -p ${PID}
        
        # Show the last few log entries
        echo -e "\nLast 10 log entries:"
        tail -n 10 "${SCHEDULER_LOG}" 2>/dev/null || echo "No logs found."
        
        # Check when the last update completed
        LAST_COMPLETED=$(grep "Scheduled update completed" "${SCHEDULER_LOG}" | tail -n 1)
        if [ -n "${LAST_COMPLETED}" ]; then
            echo -e "\nLast update completed: ${LAST_COMPLETED}"
        fi
        
        return 0
    elif [ -f "${PID_FILE}" ]; then
        echo "Scheduler is not running but PID file exists."
        echo "Clean up with: $0 cleanup"
        return 1
    else
        echo "Scheduler is not running."
        return 2
    fi
}

# Function to clean up stale PID files
cleanup() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ! ps -p ${PID} > /dev/null; then
            echo "Removing stale PID file..."
            rm -f "${PID_FILE}"
        else
            echo "PID file points to a running process. Stop the scheduler first."
            return 1
        fi
    fi
    echo "Cleanup completed."
    return 0
}

# Function to run a single update manually
run_update() {
    echo "Running manual data update..."
    cd "${APP_ROOT}" && poetry run python scripts/scheduled_update.py refresh
    return $?
}

# Function to show recent logs
show_logs() {
    local lines=${1:-50}
    if [ -f "${SCHEDULER_LOG}" ]; then
        echo "Showing last ${lines} lines of scheduler logs:"
        tail -n ${lines} "${SCHEDULER_LOG}"
    else
        echo "Scheduler log file not found: ${SCHEDULER_LOG}"
        return 1
    fi
    return 0
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
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

exit $? 