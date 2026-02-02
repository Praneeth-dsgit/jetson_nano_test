"""
Prediction endpoints for real-time and historical prediction data.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import List, Optional
import asyncio
import json
from api.models.schemas import PredictionResponse
from api.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()


@router.get("/realtime/{device_id}", response_model=PredictionResponse)
async def get_realtime_prediction(device_id: str):
    """
    Get latest real-time prediction for a device.
    
    Args:
        device_id: Device ID (e.g., "001", "1")
        
    Returns:
        Latest prediction data
    """
    # Normalize device_id to 3-digit format
    device_id = device_id.zfill(3)
    
    prediction = prediction_service.get_latest_prediction(device_id)
    
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction data found for device {device_id}"
        )
    
    return prediction


@router.get("/realtime/all")
async def get_all_realtime_predictions():
    """
    Get latest real-time predictions for all active devices.
    
    Returns:
        List of all active predictions
    """
    predictions = prediction_service.get_all_active_predictions()
    return {"predictions": predictions, "count": len(predictions)}


@router.get("/history/{device_id}")
async def get_prediction_history(device_id: str, limit: int = 10):
    """
    Get session summaries (historical data) for a device.
    
    Args:
        device_id: Device ID
        limit: Maximum number of summaries to return
        
    Returns:
        List of session summaries
    """
    device_id = device_id.zfill(3)
    summaries = prediction_service.get_session_summaries(device_id, limit)
    
    return {
        "device_id": device_id,
        "summaries": summaries,
        "count": len(summaries)
    }


@router.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """
    WebSocket endpoint for real-time prediction updates.
    
    Args:
        websocket: WebSocket connection
        device_id: Device ID
    """
    await websocket.accept()
    device_id = device_id.zfill(3)
    
    try:
        while True:
            # Get latest prediction
            prediction = prediction_service.get_latest_prediction(device_id)
            
            if prediction:
                await websocket.send_json(prediction)
            else:
                await websocket.send_json({
                    "error": "No prediction data available",
                    "device_id": device_id
                })
            
            # Update every second
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for device {device_id}")
    except Exception as e:
        print(f"WebSocket error for device {device_id}: {e}")
        await websocket.close()


@router.websocket("/ws/all")
async def websocket_all_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for all active devices.
    
    Sends updates for all active devices.
    """
    await websocket.accept()
    
    try:
        while True:
            # Get all active predictions
            predictions = prediction_service.get_all_active_predictions()
            
            await websocket.send_json({
                "timestamp": None,  # Will be set by client
                "predictions": predictions,
                "count": len(predictions)
            })
            
            # Update every second
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print("WebSocket disconnected for all devices endpoint")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

