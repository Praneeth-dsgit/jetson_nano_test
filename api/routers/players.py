"""
Player management endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from api.models.schemas import PlayerInfo, PlayerListResponse
from api.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()


@router.get("/list", response_model=PlayerListResponse)
async def list_players():
    """
    Get list of all active players.
    
    Returns:
        List of players with their current status
    """
    try:
        players_data = prediction_service.get_active_players()
        
        players = []
        for player_data in players_data:
            try:
                # Ensure all required fields are present
                player_id = str(player_data.get("player_id", "unknown"))
                device_id = str(player_data.get("device_id", "unknown"))
                status_val = player_data.get("status", "unknown")
                
                # Only create PlayerInfo if we have valid data
                if player_id != "unknown" and device_id != "unknown":
                    player_info = PlayerInfo(
                        player_id=player_id,
                        device_id=device_id,
                        status=status_val,
                        last_update=player_data.get("last_update"),
                        profile=player_data.get("profile"),
                        latest_metrics=player_data.get("latest_metrics")
                    )
                    players.append(player_info)
            except Exception as e:
                # Log error but continue processing other players
                print(f"Error processing player data: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return PlayerListResponse(players=players, total=len(players))
    except Exception as e:
        # Return empty list on error instead of crashing
        print(f"Error getting active players: {e}")
        import traceback
        traceback.print_exc()
        return PlayerListResponse(players=[], total=0)


@router.get("/{player_id}", response_model=PlayerInfo)
async def get_player_info(player_id: str):
    """
    Get detailed information for a specific player.
    
    Args:
        player_id: Player ID
        
    Returns:
        Player information
    """
    try:
        players_data = prediction_service.get_active_players()
        
        # Find player by ID
        for player_data in players_data:
            if str(player_data.get("player_id")) == str(player_id):
                return PlayerInfo(
                    player_id=str(player_data.get("player_id", "unknown")),
                    device_id=str(player_data.get("device_id", "unknown")),
                    status=player_data.get("status", "unknown"),
                    last_update=player_data.get("last_update"),
                    profile=player_data.get("profile"),
                    latest_metrics=player_data.get("latest_metrics")
                )
        
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting player info: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving player information: {str(e)}")


@router.get("/{player_id}/metrics")
async def get_player_metrics(player_id: str):
    """
    Get current health metrics for a player.
    
    Args:
        player_id: Player ID
        
    Returns:
        Current health metrics
    """
    try:
        players_data = prediction_service.get_active_players()
        
        # Find player by ID
        for player_data in players_data:
            if str(player_data.get("player_id")) == str(player_id):
                latest_prediction = prediction_service.get_latest_prediction(
                    str(player_data.get("device_id", "001"))
                )
                
                if latest_prediction:
                    return {
                        "player_id": player_id,
                        "device_id": player_data.get("device_id"),
                        "metrics": {
                            "heart_rate": latest_prediction.get("heart_rate"),
                            "stress_percent": latest_prediction.get("stress_percent"),
                            "avg_stress": latest_prediction.get("avg_stress"),
                            "stress": latest_prediction.get("stress"),
                            "vo2_max": latest_prediction.get("vo2_max"),
                            "total_trimp": latest_prediction.get("total_trimp"),
                            "current_trimp": latest_prediction.get("current_trimp"),
                            "g_impact_count": latest_prediction.get("g_impact_count"),
                            "distance": latest_prediction.get("distance"),
                            "velocity": latest_prediction.get("velocity")
                        },
                        "timestamp": latest_prediction.get("timestamp")
                    }
        
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found or no metrics available")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting player metrics: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving player metrics: {str(e)}")

