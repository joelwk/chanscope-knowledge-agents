#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Replit lightweight startup (non-blocking)...${NC}"

# Ensure local packages take precedence over system packages
export PYTHONPATH="$PWD/.pythonlibs/lib/python3.11/site-packages:${PYTHONPATH:-}"

# Create essential directories only (minimal setup)
echo -e "${YELLOW}Creating essential directories...${NC}"
mkdir -p "$PWD/logs"
mkdir -p "$PWD/temp_files"
mkdir -p "$PWD/data"
mkdir -p "$PWD/data/stratified"
mkdir -p "$PWD/scripts/utils"
touch "$PWD/scripts/utils/__init__.py"

echo -e "${GREEN}Essential directories created.${NC}"

# Disable automatic data processing for deployment
export AUTO_PROCESS_DATA_ON_INIT=false
export SKIP_DATA_UPDATE=true
export AUTO_CHECK_DATA=false

# Start background initialization only AFTER server is running
(
    echo -e "${YELLOW}Starting background initialization (will run after server starts)...${NC}"
    
    # Wait for server to be fully started before doing heavy work
    sleep 30
    
    # Now do the heavy initialization tasks in background
    echo -e "${YELLOW}Initializing PostgreSQL schema in background...${NC}"
    if [ -n "$DATABASE_URL" ] || [ -n "$PGHOST" ]; then
        python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from config.replit import PostgresDB
    db = PostgresDB()
    db.initialize_schema()
    print('PostgreSQL schema initialized successfully')
except Exception as e:
    print(f'Error initializing PostgreSQL schema: {e}')
" 2>&1 || echo -e "${YELLOW}Schema initialization skipped or failed${NC}"
    fi
    
    # Check if we need data processing (but don't do it automatically)
    echo -e "${GREEN}Background initialization completed at $(date)${NC}"
    echo "Background initialization completed at $(date)" > "$PWD/data/.replit_startup_complete"
    
) > "$PWD/logs/replit_startup_background.log" 2>&1 &

BACKGROUND_PID=$!
echo -e "${GREEN}Background tasks started with PID: $BACKGROUND_PID${NC}"

# Quick exit - let the server start immediately
echo -e "${GREEN}Startup script completed. Server can now start immediately!${NC}"
exit 0