# Replit Deployment Fix Guide

## 🔧 Root Cause & Solution

**Primary Issue**: The initialization script was running **before** the server started, blocking port binding and causing health check timeouts.

**Solution**: Restructured the startup sequence to start the server first, then run initialization in the background.

## 📁 Files Modified

### 1. `.replit` - Fixed Startup Sequence
**Before**: 
```bash
run = "bash scripts/replit_init.sh && echo 'Starting Uvicorn server...' && python -m uvicorn api.app:app --host 0.0.0.0 --port 80 --log-level info"
```

**After**:
```bash
run = "python -m uvicorn api.app:app --host 0.0.0.0 --port 80 --log-level info & SERVER_PID=$! && sleep 2 && bash scripts/replit_init.sh > logs/init.log 2>&1 & wait $SERVER_PID"
```

**Key Changes**:
- Server starts immediately in background (`&`)
- 2-second delay ensures server binds to port
- Initialization runs in background with logging
- `wait $SERVER_PID` keeps the main process alive

### 2. `scripts/replit_init.sh` - Lightweight Initialization
**Key Improvements**:
- ✅ **Fast essential setup** (directories, dependency checks)
- ✅ **Background heavy operations** (pip installs, database schema)
- ✅ **10-second server stabilization delay**
- ✅ **Optional data processing** (controlled by `AUTO_PROCESS_DATA_ON_INIT`)

### 3. `scripts/replit_setup.sh` - Pre-deployment Verification
**Purpose**: Lightweight verification script for deployment readiness
- Validates requirements.txt format
- Checks FastAPI app structure
- Verifies port configuration
- Creates default .env with safe settings

## 🚀 Deployment Steps

### Step 1: Pre-deployment Check
```bash
bash scripts/replit_setup.sh
```
This verifies your deployment is ready.

### Step 2: Environment Variables
Set these in your Replit Deployment **Secrets**:
```bash
# Database (Required for production)
DATABASE_URL=your_postgresql_url
# OR individual parameters:
PGHOST=your_host
PGUSER=your_user  
PGPASSWORD=your_password

# AWS S3 (Optional, for external data)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your_bucket

# Data Processing Control (Recommended settings)
AUTO_CHECK_DATA=false
AUTO_PROCESS_DATA_ON_INIT=false
FORCE_DATA_REFRESH=false
SKIP_EMBEDDINGS=false
ENABLE_DATA_SCHEDULER=false
```

### Step 3: Deploy
1. Click **Deploy** in Replit
2. Select **Autoscale** deployment
3. Choose appropriate machine power
4. Monitor deployment logs

### Step 4: Post-deployment Data Initialization
After successful deployment, trigger data processing:
```bash
curl https://your-app.replit.app/trigger-data-processing
```

Or use the API endpoint:
```bash
curl https://your-app.replit.app/api/v1/force-initialization
```

## 🎯 Health Check Optimization

### Immediate Response Endpoints
- **`/`** - Ultra-fast health check (middleware-intercepted)
- **`/healthz`** - Standard health check with timestamp

### Health Check Settings
```ini
[deployment.healthCheck]
path = "/"
port = 80
initialDelay = 10    # Reduced from 30
timeout = 5          # Standard timeout
period = 10          # Check every 10 seconds
consecutiveFailures = 3  # Less tolerant (faster detection)
```

## 🔍 Monitoring & Troubleshooting

### Check Deployment Status
```bash
# Check if server is responding
curl https://your-app.replit.app/

# Check initialization status
curl https://your-app.replit.app/api/v1/initialization-status
```

### Monitor Logs
- **Deployment logs**: Available in Replit deployment dashboard
- **Initialization logs**: `logs/replit_init_background.log`
- **Data processing logs**: `logs/data_processing.log`

### Common Issues & Solutions

#### Issue: "Port not opening in time"
**Solution**: The new startup sequence should fix this. If it persists:
1. Check that `python -m uvicorn api.app:app --host 0.0.0.0 --port 80` works locally
2. Verify no dependencies are missing in `requirements.txt`

#### Issue: "Health check failing"
**Solution**: 
1. Test the root endpoint: `curl https://your-app.replit.app/`
2. Check if middleware is working properly
3. Verify no blocking operations in the root endpoint

#### Issue: "Database connection errors"
**Solution**:
1. Verify `DATABASE_URL` is set in deployment secrets
2. Check PostgreSQL database is accessible from Replit
3. Use `/api/v1/initialization-status` to see specific errors

## 🔄 Data Processing Flow

### Background Initialization (Automatic)
1. Server starts and binds to port 80 (immediate)
2. Health checks pass (within 10 seconds)
3. Background init waits 10 seconds for server stability
4. Dependencies install (if needed)
5. Database schema initialization
6. Service verification (AWS, Replit KV)

### Data Processing (Manual Trigger)
1. Call `/trigger-data-processing` endpoint
2. Data fetching from AWS S3
3. Stratified sampling
4. Embedding generation (if not skipped)
5. Storage in PostgreSQL + Replit KV

## ✅ Success Indicators

### Deployment Success
- ✅ Health check passes within 10 seconds
- ✅ Root endpoint (`/`) returns `{"status": "ok"}`
- ✅ No "port not opening" errors

### Initialization Success  
- ✅ Background logs show "Background initialization completed successfully!"
- ✅ `/api/v1/initialization-status` shows services are available
- ✅ Database schema is created

### Data Processing Success
- ✅ `/api/v1/initialization-status` shows `"ready": true`
- ✅ Query endpoints return results
- ✅ Data files exist or database has records

## 🎛️ Configuration Options

### Environment Variables for Fine-tuning
```bash
# Startup behavior
AUTO_CHECK_DATA=false                    # Don't auto-check data on startup
AUTO_PROCESS_DATA_ON_INIT=false         # Don't auto-process data on init

# Data processing
FORCE_DATA_REFRESH=false                # Don't force refresh unless needed
SKIP_EMBEDDINGS=false                   # Generate embeddings (slower but better results)

# Scheduler
ENABLE_DATA_SCHEDULER=false             # Disable periodic updates initially
DATA_UPDATE_INTERVAL=3600               # Update every hour (if enabled)

# Performance
FASTAPI_DEBUG=false                     # Disable debug mode for faster startup
```

## 🆘 Emergency Rollback

If deployment fails completely:

1. **Revert `.replit` run command**:
   ```bash
   run = "python -m uvicorn api.app:app --host 0.0.0.0 --port 80 --log-level info"
   ```

2. **Disable all background processing**:
   ```bash
   AUTO_CHECK_DATA=false
   AUTO_PROCESS_DATA_ON_INIT=false
   ENABLE_DATA_SCHEDULER=false
   ```

3. **Use minimal initialization**:
   ```bash
   run = "mkdir -p logs data temp_files && python -m uvicorn api.app:app --host 0.0.0.0 --port 80 --log-level info"
   ```

This should get a basic server running for debugging. 