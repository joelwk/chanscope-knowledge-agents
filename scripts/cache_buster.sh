#!/bin/bash
# Cache Buster Script - Forces Replit to rebuild all cached layers
# Run this before deployment to ensure Python 3.11 environment

echo "🚀 CACHE BUSTER: Forcing complete environment rebuild"
echo "This will invalidate all cached layers and force Python 3.11 rebuild"

# 1. Add timestamp to force cache invalidation
TIMESTAMP=$(date +%s)
echo "# Cache buster timestamp: $TIMESTAMP" >> replit.nix

# 2. Remove all local caches
rm -rf .venv venv ENV env
rm -rf ~/.cache/pip
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 3. Verify system Python
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "System Python version: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "❌ ERROR: System Python is $PYTHON_VERSION, not 3.11"
    echo "Please wait for Nix environment to rebuild and try again"
    exit 1
fi

echo "✅ Cache buster complete. Deploy now to force fresh Python 3.11 build." 