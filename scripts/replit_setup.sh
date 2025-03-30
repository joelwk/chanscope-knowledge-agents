#!/bin/bash
# Setup script for Replit environment
set -e

echo "Starting Replit environment setup..."

# Create necessary directories
mkdir -p data/stratified
mkdir -p data/logs
mkdir -p temp_files
mkdir -p logs

# Set environment variables
echo "Setting environment variables for Replit..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file..."
  cat > .env << EOF
# Environment settings
REPLIT_ENV=replit
DEPLOYMENT_ENV=replit
API_PORT=80
API_HOST=0.0.0.0
LOG_LEVEL=INFO

# Data settings
ROOT_DATA_PATH=data
STRATIFIED_PATH=data/stratified
PATH_TEMP=temp_files
DATA_RETENTION_DAYS=14
FILTER_DATE=
SAMPLE_SIZE=1000
EMBEDDING_BATCH_SIZE=10
MAX_WORKERS=2

# Feature flags
AUTO_CHECK_DATA=true
FORCE_DATA_REFRESH=false
SKIP_EMBEDDINGS=false
EOF
fi

# Check if replit packages are installed
echo "Checking for required packages..."
if ! pip show replit > /dev/null; then
  echo "Installing replit package..."
  pip install replit
fi

if ! pip show psycopg2-binary > /dev/null; then
  echo "Installing psycopg2-binary package..."
  pip install psycopg2-binary
fi

# Check for Replit DB integration
echo "Checking Replit database environment..."

# Handle PostgreSQL environment variables (from Replit Secrets or environment)
if [[ -n "$REPLIT_DB_URL" ]]; then
  echo "Replit Key-Value store URL found"
else
  echo "WARNING: REPLIT_DB_URL not found, Key-Value store operations may fail"
fi

# Check for standard PostgreSQL variables first
if [[ -n "$DATABASE_URL" ]]; then
  echo "DATABASE_URL environment variable found"
  POSTGRES_CONNECTION_FOUND=true
elif [[ -n "$PGHOST" && -n "$PGUSER" && -n "$PGPASSWORD" && -n "$PGDATABASE" ]]; then
  echo "PostgreSQL connection parameters found (PGHOST, PGUSER, etc.)"
  POSTGRES_CONNECTION_FOUND=true
elif [[ -n "$POSTGRES_URL" && -n "$POSTGRES_USER" && -n "$POSTGRES_PASSWORD" && -n "$POSTGRES_DATABASE" ]]; then
  echo "PostgreSQL connection parameters found (POSTGRES_URL, POSTGRES_USER, etc.)"
  # Map old-style variables to the standard ones expected by psycopg2
  export PGHOST="$POSTGRES_URL"
  export PGUSER="$POSTGRES_USER"
  export PGPASSWORD="$POSTGRES_PASSWORD"
  export PGDATABASE="$POSTGRES_DATABASE"
  POSTGRES_CONNECTION_FOUND=true
else
  echo "WARNING: PostgreSQL connection details not found, skipping schema initialization"
  echo "To initialize schema, set DATABASE_URL or PGHOST, PGUSER, PGPASSWORD, and PGDATABASE environment variables"
  POSTGRES_CONNECTION_FOUND=false
fi

# Run database schema initialization if PostgreSQL is available
if [[ "$POSTGRES_CONNECTION_FOUND" == true ]]; then
  echo "Initializing PostgreSQL schema..."
  # Run schema initialization in a Python script with proper error handling
  python - << EOF
import sys
try:
    from config.replit import PostgresDB
    db = PostgresDB()
    db.initialize_schema()
    print('Schema initialized successfully')
    sys.exit(0)
except Exception as e:
    print(f'Error initializing schema: {e}')
    sys.exit(1)
EOF

  if [ $? -ne 0 ]; then
    echo "WARNING: Schema initialization failed. Application may not work correctly."
  else
    echo "PostgreSQL schema initialized successfully"
  fi
fi

echo "Replit environment setup complete!"
echo "You can now run the application with: python -m api.app" 