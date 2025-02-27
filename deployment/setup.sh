#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Determine if we're in a Replit environment
if [ -n "$REPL_ID" ] || [ "$REPLIT_ENV" = "replit" ] || [ "$REPLIT_ENV" = "true" ] || [ "$REPLIT_ENV" = "production" ]; then
    IS_REPLIT=true
    echo "Detected Replit environment (REPL_ID: $REPL_ID, REPL_SLUG: $REPL_SLUG)"
else
    IS_REPLIT=false
    echo "Not in Replit environment, assuming Docker or local"
fi

# Determine application root directory based on environment
if [ "$IS_REPLIT" = true ]; then
    # In Replit, use the workspace directory
    APP_ROOT="$PWD"
    DATA_DIR="$APP_ROOT/data"
    echo "Running in Replit environment. App root: $APP_ROOT"
else
    # In Docker or local, use /app if it exists and is writable, otherwise use current directory
    if [ -d "/app" ] && [ -w "/app" ]; then
        APP_ROOT="/app"
        DATA_DIR="/app/data"
        echo "Running in Docker environment. App root: $APP_ROOT"
    else
        APP_ROOT="$PWD"
        DATA_DIR="$APP_ROOT/data"
        echo "Running in local environment. App root: $APP_ROOT"
    fi
fi

cd "$APP_ROOT"
echo "Changed to directory: $(pwd)"

# Function to clean environment variables
clean_env_vars() {
    env_file="$1"
    echo "Loading environment variables from $env_file..."
    
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ -n "$line" && ! "$line" =~ ^# ]]; then
            cleaned_line=$(echo "$line" | tr -d '\r' | sed 's/[[:space:]]*$//')
            
            if [[ "$cleaned_line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                var_name="${BASH_REMATCH[1]}"
                var_value="${BASH_REMATCH[2]}"
                
                var_value="${var_value#\"}"
                var_value="${var_value%\"}"
                var_value="${var_value#\'}"
                var_value="${var_value%\'}"
                
                case "$var_name" in
                    "LOG_LEVEL")
                        var_value=$(echo "$var_value" | tr '[:upper:]' '[:lower:]')
                        ;;
                    "OPENAI_API_KEY"|"GROK_API_KEY"|"VENICE_API_KEY"|"AWS_ACCESS_KEY_ID"|"AWS_SECRET_ACCESS_KEY"|"AWS_DEFAULT_REGION"|"S3_BUCKET")
                        var_value=$(echo "$var_value" | tr -d '\r')
                        ;;
                    *)
                        var_value=$(echo "$var_value" | tr -d '\r\n' | tr '[:upper:]' '[:lower:]')
                        ;;
                esac
                
                # Only export if the variable isn't already set
                if [ -z "${!var_name}" ]; then
                    export "$var_name=$var_value"
                    echo "Exported: $var_name"
                fi
            fi
        fi
    done < "$env_file"
}

# Load environment variables, trying .env in multiple locations
if [ "$ENVIRONMENT_LOADED" != "true" ]; then
    if [ -f "$APP_ROOT/.env" ]; then
        clean_env_vars "$APP_ROOT/.env"
    elif [ -f ".env" ]; then
        clean_env_vars ".env"
    else
        echo "No .env file found"
    fi
    export ENVIRONMENT_LOADED=true
fi

# Set default environments if not set
export ENVIRONMENT="${ENVIRONMENT:-development}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
# Convert LOG_LEVEL to lowercase for uvicorn compatibility
LOG_LEVEL=$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')
export API_PORT="${API_PORT:-80}"
export HOST="${HOST:-0.0.0.0}"
export API_WORKERS="${API_WORKERS:-4}"
export WORKER_ID="Spawn1"

# Set Replit-specific environment variables if in Replit
if [ "$IS_REPLIT" = true ]; then
    export REPLIT_ENV="${REPLIT_ENV:-replit}"
    export FASTAPI_ENV="${FASTAPI_ENV:-production}"
    export PORT="80"  # Replit expects port 80
    export API_PORT="80"
    echo "Set Replit environment variables"
fi

# Verify required environment variables
echo "Verifying environment configuration..."
required_vars=(
    "OPENAI_API_KEY"
    "API_PORT"
    "LOG_LEVEL"
    "AWS_DEFAULT_REGION"
    "AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY"
    "S3_BUCKET"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -n "${!var}" ]; then
        echo "Verified: $var"
    else
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "Warning: Missing recommended environment variables:"
    printf '%s\n' "${missing_vars[@]}"
    # Not exiting with error, as some variables might be optional in certain environments
fi

# Print AWS configuration for debugging
echo "AWS Configuration:"
echo "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
echo "S3_BUCKET=${S3_BUCKET}"
echo "AWS_ACCESS_KEY_ID is set: $([ -n "$AWS_ACCESS_KEY_ID" ] && echo "yes" || echo "no")"
echo "AWS_SECRET_ACCESS_KEY is set: $([ -n "$AWS_SECRET_ACCESS_KEY" ] && echo "yes" || echo "no")"

# Create data directory if it doesn't exist
echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Clean up any stale initialization markers
echo "Cleaning up any stale initialization markers..."
rm -f "$DATA_DIR/.initialization_in_progress"
rm -f "$DATA_DIR/.initialization_state"
rm -f "$DATA_DIR/.worker.lock"
rm -f "$DATA_DIR/.worker.lock."*
rm -f "$DATA_DIR/.primary_worker"

# Create setup complete marker
echo "Creating setup complete marker"
touch "$DATA_DIR/.setup_complete"
echo "Setup completed successfully"

# Start the FastAPI application using Uvicorn with appropriate configuration
echo "Starting FastAPI application..."
echo "Host: $HOST, Port: $API_PORT, Workers: $API_WORKERS, Log level: $LOG_LEVEL"
echo "Environment: $ENVIRONMENT, Replit: $IS_REPLIT"

# In Replit, use the app instance from api/app.py which has Replit-specific configurations
if [ "$IS_REPLIT" = true ]; then
    echo "Starting with Replit-specific configuration"
    exec poetry run python -m uvicorn api.app:app --host "$HOST" --port "$API_PORT" --log-level "$LOG_LEVEL"
else
    # In Docker/local, use the standard approach
    echo "Starting with standard configuration"
    exec uvicorn api:app --host "$HOST" --port "$API_PORT" --workers "$API_WORKERS" --log-level "$LOG_LEVEL"
fi