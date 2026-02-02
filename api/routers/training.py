"""
Training management endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import subprocess
import os
import sys
from api.models.schemas import TrainingStatus

router = APIRouter()


class TrainingRequest(BaseModel):
    """Training request model."""
    player_ids: Optional[List[int]] = None
    force: bool = False


@router.post("/start", response_model=TrainingStatus)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start ML model training.
    
    Args:
        request: Training request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Training status
    """
    # Check if training can start (no prediction running)
    prediction_lockfile = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.prediction_running.lock')
    
    if os.path.exists(prediction_lockfile) and not request.force:
        raise HTTPException(
            status_code=409,
            detail="Prediction is currently running. Cannot start training. Use force=true to override."
        )
    
    # Check if training is already running
    training_lockfile = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.training_running.lock')
    
    if os.path.exists(training_lockfile):
        return TrainingStatus(
            is_running=True,
            status="running",
            message="Training is already running"
        )
    
    # Set environment variable if force is requested
    if request.force:
        os.environ["FORCE_TRAINING"] = "true"
    
    # Start training in background
    background_tasks.add_task(run_training_task)
    
    return TrainingStatus(
        is_running=True,
        status="started",
        message="Training process initiated"
    )


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """
    Get current training status.
    
    Returns:
        Training status information
    """
    training_lockfile = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.training_running.lock')
    
    is_running = os.path.exists(training_lockfile)
    
    if is_running:
        # Try to read PID from lockfile
        try:
            with open(training_lockfile, 'r') as f:
                pid = f.read().strip()
                return TrainingStatus(
                    is_running=True,
                    status="running",
                    message=f"Training is running (PID: {pid})"
                )
        except:
            return TrainingStatus(
                is_running=True,
                status="running",
                message="Training is running"
            )
    
    return TrainingStatus(
        is_running=False,
        status="idle",
        message="No training in progress"
    )


@router.post("/stop")
async def stop_training():
    """
    Stop running training (if possible).
    
    Note: This attempts to gracefully stop training by removing the lockfile.
    The actual training process should handle SIGTERM/SIGINT properly.
    """
    training_lockfile = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.training_running.lock')
    
    if os.path.exists(training_lockfile):
        try:
            # Read PID
            with open(training_lockfile, 'r') as f:
                pid = f.read().strip()
            
            # Try to stop the process
            try:
                import signal
                os.kill(int(pid), signal.SIGTERM)
            except (ValueError, ProcessLookupError, OSError):
                pass
            
            # Remove lockfile
            os.remove(training_lockfile)
            
            return {"status": "stopped", "message": "Training stop signal sent"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error stopping training: {e}")
    
    return {"status": "idle", "message": "No training was running"}


def run_training_task():
    """Run training script in background."""
    try:
        # Resolve training script path
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        training_script = os.path.join(current_dir, 'training', 'sup_ml_rf_training.py')
        
        if not os.path.exists(training_script):
            print(f"Training script not found: {training_script}")
            return
        
        # Run training script
        subprocess.run([sys.executable, training_script], check=False)
    except Exception as e:
        print(f"Error running training: {e}")

