#!/bin/bash
# Setup script for Replit environment (pip based)
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
  echo "Creating .env file from template..."
  cat > .env << EOF
# Environment settings
REPLIT_ENV=replit
DEPLOYMENT_ENV=replit
API_PORT=80
API_HOST=0.0.0.0
LOG_LEVEL=INFO
# Add other necessary env vars here
EOF
fi

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "Warning: requirements.txt not found."
fi

# Check for Replit DB integration
echo "Checking Replit database environment..."
if [[ -z "$REPLIT_DB_URL" ]]; then
  echo "WARNING: REPLIT_DB_URL not found, Key-Value store operations may fail"
fi

# Check for standard PostgreSQL variables
if [[ -z "$DATABASE_URL" ]] && [[ -z "$PGHOST" ]]; then
  echo "WARNING: PostgreSQL connection details not found, skipping schema initialization"
else
  echo "PostgreSQL connection details found. Initializing schema..."
  python3 -c "
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
"
  if [ $? -ne 0 ]; then
    echo "WARNING: Schema initialization failed. Application may not work correctly."
  else
    echo "PostgreSQL schema initialized successfully"
  fi
fi

echo "Replit environment setup complete!"
echo "You can now run the application with: python3 -m uvicorn api.app:app" 