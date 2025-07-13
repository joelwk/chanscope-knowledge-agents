#!/bin/bash
# Lightweight setup verification script for Replit deployment
set -e

echo "Starting Replit deployment verification..."

# Create necessary directories if they don't exist
echo "Ensuring directory structure..."
mkdir -p data/stratified
mkdir -p logs
mkdir -p temp_files

# Check Python version
echo "Checking Python version..."
python3 --version

# Verify critical dependencies are installable (don't install them here)
echo "Verifying requirements.txt format..."
if [ -f "requirements.txt" ]; then
  echo "✓ requirements.txt found"
  # Just validate the format, don't install
  python3 -c "
import pkg_resources
try:
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    pkg_resources.Requirement.parse(line)
                except Exception as e:
                    print(f'Invalid requirement: {line} - {e}')
                    exit(1)
    print('Requirements format is valid')
except Exception as e:
    print(f'Error reading requirements.txt: {e}')
    exit(1)
"
else
  echo "❌ requirements.txt not found!"
  exit 1
fi

# Check environment variable configuration
echo "Checking environment configuration..."

# Create .env file with defaults if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating default .env file..."
  cat > .env << EOF
# Environment settings for Replit deployment
REPLIT_ENV=replit
DEPLOYMENT_ENV=replit
API_PORT=80
API_HOST=0.0.0.0
LOG_LEVEL=INFO

# Data processing control (disabled by default for fast startup)
AUTO_CHECK_DATA=false
AUTO_PROCESS_DATA_ON_INIT=false
FORCE_DATA_REFRESH=false
SKIP_EMBEDDINGS=false

# Scheduler (disabled by default)
ENABLE_DATA_SCHEDULER=false
DATA_UPDATE_INTERVAL=3600
EOF
  echo "✓ Created default .env file"
else
  echo "✓ .env file exists"
fi

# Verify FastAPI app can be imported (lightweight check)
echo "Verifying FastAPI app structure..."
python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    # Just check if the app module exists and can be imported
    from api.app import app
    print('✓ FastAPI app can be imported')
    
    # Check if it has the required structure
    if hasattr(app, 'include_router'):
        print('✓ FastAPI app structure is valid')
    else:
        print('❌ FastAPI app missing required structure')
        sys.exit(1)
        
except ImportError as e:
    print(f'❌ Cannot import FastAPI app: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error with FastAPI app: {e}')
    sys.exit(1)
"

# Check database configuration (don't actually connect)
echo "Checking database configuration..."
if [[ -n "$DATABASE_URL" ]] || [[ -n "$PGHOST" ]]; then
  echo "✓ PostgreSQL connection variables found"
else
  echo "⚠️  PostgreSQL connection variables not found"
  echo "   This is OK for testing, but production deployments need database access"
fi

if [[ -n "$REPLIT_DB_URL" ]]; then
  echo "✓ Replit Key-Value store URL found"
else
  echo "⚠️  REPLIT_DB_URL not found"
  echo "   This is normal in development, but may be needed for production"
fi

# Verify port configuration
echo "Checking port configuration..."
if grep -q "localPort = 80" .replit && grep -q "externalPort = 80" .replit; then
  echo "✓ Port configuration is correct for deployment"
else
  echo "❌ Port configuration may be incorrect"
  echo "   Check .replit file for proper port settings"
fi

# Check run command
echo "Checking deployment run command..."
if grep -q "python -m uvicorn api.app:app --host 0.0.0.0 --port 80" .replit; then
  echo "✓ Run command starts server properly"
else
  echo "⚠️  Run command may need verification"
fi

# Check health check endpoint
echo "Verifying health check endpoint..."
python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from api.app import app
    
    # Check if the root endpoint is defined
    routes = [route.path for route in app.routes]
    if '/' in routes:
        print('✓ Health check endpoint (/) is available')
    else:
        print('❌ Root health check endpoint not found')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Error checking health endpoint: {e}')
    sys.exit(1)
"

echo ""
echo "✅ Replit deployment verification completed successfully!"
echo ""
echo "📋 Deployment checklist:"
echo "   ✓ Directory structure created"
echo "   ✓ Requirements format validated"  
echo "   ✓ FastAPI app structure verified"
echo "   ✓ Health check endpoint confirmed"
echo "   ✓ Port configuration checked"
echo ""
echo "🚀 Ready for deployment!"
echo ""
echo "💡 To deploy:"
echo "   1. Ensure all secrets are set in Replit Deployment config"
echo "   2. Click Deploy in Replit"
echo "   3. Monitor deployment logs for any issues"
echo "   4. Use /trigger-data-processing endpoint after deployment to initialize data" 