#!/usr/bin/env python
"""
Web Dashboard for Automated Refresh System
Provides a web interface for monitoring and controlling data refreshes
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add the project root to the Python path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Import the automated refresh manager
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from automated_refresh import AutomatedRefreshManager, RefreshStatus

app = FastAPI(title="ChanScope Refresh Dashboard", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global refresh manager instance
refresh_manager: Optional[AutomatedRefreshManager] = None
CONTROL_TOKEN: Optional[str] = os.environ.get("REFRESH_CONTROL_TOKEN") or os.environ.get("DASHBOARD_CONTROL_TOKEN")

def _auth_ok(request: Request) -> bool:
    """Validate shared secret if configured.
    Accepts header X-Refresh-Token or Authorization: Bearer <token>, or query param token.
    If no CONTROL_TOKEN configured, allow by default.
    """
    if not CONTROL_TOKEN:
        return True
    token = request.headers.get("x-refresh-token")
    if not token:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
    if not token:
        token = request.query_params.get("token")
    return token == CONTROL_TOKEN

class RefreshConfig(BaseModel):
    interval_seconds: int = 3600
    max_retries: int = 3

class RefreshCommand(BaseModel):
    action: str  # "start", "stop", "refresh_once"

# HTML Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChanScope Refresh Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            color: #2d3748;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #718096;
            font-size: 1.1rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .card-title {
            font-size: 1.2rem;
            color: #4a5568;
            margin-bottom: 15px;
            font-weight: 600;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-running { background: #48bb78; }
        .status-idle { background: #cbd5e0; }
        .status-error { background: #f56565; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .metric:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        .metric-label {
            color: #718096;
        }
        
        .metric-value {
            font-weight: 600;
            color: #2d3748;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .btn-primary { background: #667eea; }
        .btn-success { background: #48bb78; }
        .btn-danger { background: #f56565; }
        .btn-warning { background: #ed8936; }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
        
        .log-container {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .log-entry {
            margin-bottom: 5px;
        }
        
        .log-time {
            color: #cbd5e0;
            margin-right: 10px;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        input[type="number"] {
            flex: 1;
            padding: 10px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ChanScope Automated Refresh Dashboard</h1>
            <div class="subtitle">Monitor and control data refresh operations</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2 class="card-title">System Status</h2>
                <div id="system-status">
                    <div class="metric">
                        <span class="metric-label">Status</span>
                        <span class="metric-value">
                            <span class="status-indicator status-idle"></span>
                            <span id="status-text">Idle</span>
                        </span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Current Job</span>
                        <span class="metric-value" id="current-job">None</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Next Run</span>
                        <span class="metric-value" id="next-run">Not scheduled</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Interval</span>
                        <span class="metric-value" id="interval">60 minutes</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2 class="card-title">Performance Metrics</h2>
                <div id="metrics">
                    <div class="metric">
                        <span class="metric-label">Total Runs</span>
                        <span class="metric-value" id="total-runs">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Current Rows</span>
                        <span class="metric-value" id="current-rows">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value" id="success-rate">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Duration</span>
                        <span class="metric-value" id="avg-duration">0s</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Rows</span>
                        <span class="metric-value" id="avg-rows">0</span>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="success-progress" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="card">
                <h2 class="card-title">Controls</h2>
                <div class="controls">
                    <button class="btn-success" onclick="startRefresh()">Start Auto-Refresh</button>
                    <button class="btn-danger" onclick="stopRefresh()">Stop</button>
                    <button class="btn-warning" onclick="runOnce()">Run Once</button>
                </div>
                <div class="input-group">
                    <input type="number" id="interval-input" placeholder="Interval (seconds)" min="60" value="3600">
                    <button class="btn-primary" onclick="updateInterval()">Update Interval</button>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 class="card-title">Recent Activity</h2>
            <div class="log-container" id="activity-log">
                <div class="log-entry">
                    <span class="log-time">--:--:--</span>
                    <span>Waiting for activity...</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let refreshInterval;
        // Resolve the base mount path (e.g., '/refresh' when mounted)
        const __basePath = (function () {
            const path = window.location.pathname.replace(/\/$/, '');
            return path || '';
        })();
        const apiUrl = (suffix) => (__basePath ? `${__basePath}${suffix}` : suffix);
        const tokenParam = new URLSearchParams(window.location.search).get('token');
        const withToken = (url) => tokenParam ? `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(tokenParam)}` : url;
        
        async function fetchStatus() {
            try {
                const [statusRes, metricsRes] = await Promise.all([
                    fetch(apiUrl('/api/status')),
                    fetch(apiUrl('/api/metrics'))
                ]);
                
                const status = await statusRes.json();
                const metrics = await metricsRes.json();
                
                updateDisplay(status, metrics);
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }
        
        function updateDisplay(status, metrics) {
            // Update status
            const statusIndicator = document.querySelector('.status-indicator');
            const statusText = document.getElementById('status-text');
            
            if (status.is_running) {
                statusIndicator.className = 'status-indicator status-running';
                statusText.textContent = 'Running';
            } else {
                statusIndicator.className = 'status-indicator status-idle';
                statusText.textContent = 'Idle';
            }
            
            document.getElementById('current-job').textContent = status.current_job?.id || 'None';
            document.getElementById('next-run').textContent = 
                status.next_run ? new Date(status.next_run).toLocaleString() : 'Not scheduled';
            document.getElementById('interval').textContent = 
                `${Math.round(status.interval_seconds / 60)} minutes`;
            
            // Update metrics
            document.getElementById('total-runs').textContent = metrics.total_runs || 0;
            document.getElementById('current-rows').textContent = 
                (metrics.current_row_count || 0).toLocaleString();
            document.getElementById('success-rate').textContent = 
                `${Math.round(metrics.success_rate || 0)}%`;
            document.getElementById('avg-duration').textContent = 
                `${Math.round(metrics.average_duration || 0)}s`;
            document.getElementById('avg-rows').textContent = 
                Math.round(metrics.average_rows_processed || 0).toLocaleString();
            
            // Update progress bar
            const progressBar = document.getElementById('success-progress');
            progressBar.style.width = `${metrics.success_rate || 0}%`;
            
            // Update activity log
            updateActivityLog(status, metrics);
        }
        
        function updateActivityLog(status, metrics) {
            const log = document.getElementById('activity-log');
            const now = new Date().toLocaleTimeString();
            
            let logHtml = '';
            
            if (status.current_job) {
                logHtml += `<div class="log-entry">
                    <span class="log-time">${now}</span>
                    <span>Job ${status.current_job.id} is ${status.current_job.status}</span>
                </div>`;
            }
            
            if (metrics.last_success) {
                const successTime = new Date(metrics.last_success).toLocaleString();
                logHtml += `<div class="log-entry">
                    <span class="log-time">${successTime}</span>
                    <span style="color: #48bb78;">Last successful refresh</span>
                </div>`;
            }
            
            if (metrics.last_failure) {
                const failTime = new Date(metrics.last_failure).toLocaleString();
                logHtml += `<div class="log-entry">
                    <span class="log-time">${failTime}</span>
                    <span style="color: #f56565;">Last failed refresh</span>
                </div>`;
            }
            
            if (logHtml) {
                log.innerHTML = logHtml;
            }
        }
        
        async function startRefresh() {
            const interval = document.getElementById('interval-input').value;
            const response = await fetch(withToken(apiUrl('/api/control')), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'start'})
            });
            
            if (response.ok) {
                alert('Auto-refresh started');
                fetchStatus();
            } else {
                alert('Failed to start auto-refresh');
            }
        }
        
        async function stopRefresh() {
            const response = await fetch(withToken(apiUrl('/api/control')), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'stop'})
            });
            
            if (response.ok) {
                alert('Auto-refresh stopped');
                fetchStatus();
            } else {
                alert('Failed to stop auto-refresh');
            }
        }
        
        async function runOnce() {
            const response = await fetch(withToken(apiUrl('/api/control')), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'refresh_once'})
            });
            
            if (response.ok) {
                alert('Manual refresh started');
                fetchStatus();
            } else {
                alert('Failed to start manual refresh');
            }
        }
        
        async function updateInterval() {
            const interval = document.getElementById('interval-input').value;
            const response = await fetch(withToken(apiUrl('/api/config')), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({interval_seconds: parseInt(interval), max_retries: 3})
            });
            
            if (response.ok) {
                alert(`Interval updated to ${interval} seconds`);
                fetchStatus();
            } else {
                alert('Failed to update interval');
            }
        }
        
        // Start auto-refresh of dashboard
        fetchStatus();
        refreshInterval = setInterval(fetchStatus, 5000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML"""
    return DASHBOARD_HTML

@app.get("/api/status")
async def get_status():
    """Get current refresh status"""
    if refresh_manager:
        return refresh_manager.get_status()
    
    # Try to read from file
    status_file = Path("data/refresh_status.json")
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    
    return {
        "is_running": False,
        "current_job": None,
        "next_run": None,
        "interval_seconds": 3600
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get performance metrics"""
    if refresh_manager:
        return refresh_manager.get_metrics()
    
    # Try to read from file
    metrics_file = Path("data/refresh_metrics.json")
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    return {
        "total_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "success_rate": 0,
        "average_duration": 0,
        "average_rows_processed": 0,
        "current_row_count": 0
    }

@app.post("/api/control")
async def control_refresh(command: RefreshCommand, background_tasks: BackgroundTasks, request: Request):
    """Control the refresh system"""
    global refresh_manager
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if command.action == "start":
        if not refresh_manager:
            refresh_manager = AutomatedRefreshManager()
        
        if not refresh_manager.is_running:
            background_tasks.add_task(refresh_manager.run_continuous)
            return {"status": "started"}
        return {"status": "already_running"}
    
    elif command.action == "stop":
        if refresh_manager:
            refresh_manager.is_running = False
            return {"status": "stopped"}
        return {"status": "not_running"}
    
    elif command.action == "refresh_once":
        if not refresh_manager:
            refresh_manager = AutomatedRefreshManager()
        
        background_tasks.add_task(refresh_manager.run_refresh)
        return {"status": "refresh_triggered"}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@app.post("/api/config")
async def update_config(config: RefreshConfig, request: Request):
    """Update refresh configuration"""
    global refresh_manager
    if not _auth_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not refresh_manager:
        refresh_manager = AutomatedRefreshManager(
            interval_seconds=config.interval_seconds,
            max_retries=config.max_retries
        )
    else:
        refresh_manager.interval_seconds = config.interval_seconds
        refresh_manager.max_retries = config.max_retries
    
    return {"status": "updated", "config": config.dict()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.on_event("startup")
async def startup_autostart_manager():
    """Optionally auto-start the automated refresh manager on app startup."""
    try:
        enable = os.environ.get("AUTO_REFRESH_MANAGER", "false").lower() in ("true", "1", "yes")
        interval = int(os.environ.get(
            "DATA_REFRESH_INTERVAL",
            os.environ.get(
                "DATA_UPDATE_INTERVAL",
                os.environ.get("REFRESH_INTERVAL", "3600")
            )
        ))
        if enable:
            global refresh_manager
            if not refresh_manager:
                refresh_manager = AutomatedRefreshManager(interval_seconds=interval)
            else:
                refresh_manager.interval_seconds = interval
            if not refresh_manager.is_running:
                asyncio.create_task(refresh_manager.run_continuous())
    except Exception:
        # Do not crash the app on startup issues
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
