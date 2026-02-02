"""
System status and health endpoints.
"""

from fastapi import APIRouter
import psutil
import os
from datetime import datetime
from api.models.schemas import SystemStatus

router = APIRouter()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get current system status.
    
    Returns:
        System status information including CPU, memory, disk usage,
        and process status
    """
    # Resolve project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check for lockfiles
    prediction_lockfile = os.path.join(project_root, '.prediction_running.lock')
    training_lockfile = os.path.join(project_root, '.training_running.lock')
    
    return SystemStatus(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=psutil.virtual_memory().percent,
        disk_usage=psutil.disk_usage('/').percent,
        prediction_running=os.path.exists(prediction_lockfile),
        training_running=os.path.exists(training_lockfile),
        timestamp=datetime.now().isoformat()
    )


@router.get("/health")
async def get_system_health():
    """
    Get detailed system health information.
    
    Returns:
        Detailed health metrics
    """
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent": disk.percent
        },
        "processes": {
            "prediction_running": os.path.exists(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.prediction_running.lock')
            ),
            "training_running": os.path.exists(
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.training_running.lock')
            )
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/processes")
async def get_processes():
    """
    Get information about running project-related processes.
    
    Returns:
        List of relevant processes
    """
    processes = []
    
    # Check for prediction processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline:
                cmdline_str = ' '.join(cmdline)
                if 'main.py' in cmdline_str or 'test_30_players.py' in cmdline_str:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "type": "prediction",
                        "cmdline": cmdline_str
                    })
                elif 'sup_ml_rf_training.py' in cmdline_str:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "type": "training",
                        "cmdline": cmdline_str
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return {"processes": processes, "count": len(processes)}

