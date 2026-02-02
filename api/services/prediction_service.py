"""
Service for accessing prediction data from the prediction engine outputs.
"""

import json
import os
import glob
from typing import Optional, Dict, List
from datetime import datetime
import re


class PredictionService:
    """Service to access prediction data from file system."""
    
    def __init__(self, base_output_dir: str = None):
        """
        Initialize prediction service.
        
        Args:
            base_output_dir: Base directory for prediction outputs.
                           If None, will try to resolve from project structure.
        """
        if base_output_dir is None:
            # Try to resolve from project root
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_output_dir = os.path.join(current_dir, 'data', 'prediction_outputs')
        
        self.base_output_dir = base_output_dir
    
    def _resolve_output_dir(self) -> str:
        """Resolve the prediction outputs directory path."""
        if os.path.exists(self.base_output_dir):
            return self.base_output_dir
        
        # Try alternative paths
        alt_paths = [
            os.path.join('data', 'prediction_outputs'),
            os.path.join('..', 'data', 'prediction_outputs'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prediction_outputs')
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        # Return default even if it doesn't exist yet
        return os.path.abspath(self.base_output_dir)
    
    def get_latest_prediction(self, device_id: str) -> Optional[Dict]:
        """
        Get latest prediction for a device.
        
        Args:
            device_id: Device ID (e.g., "001", "002")
            
        Returns:
            Prediction data dictionary or None if not found
        """
        output_dir = self._resolve_output_dir()
        
        # Find player folder for this device
        # Pattern: A{athlete_id}_Device_{device_id} or A{athlete_id}_{name}
        pattern = os.path.join(output_dir, f"A*_Device_*{device_id.zfill(3)}*")
        player_dirs = glob.glob(pattern)
        
        if not player_dirs:
            # Try alternative pattern
            pattern = os.path.join(output_dir, "A*")
            all_dirs = glob.glob(pattern)
            for dir_path in all_dirs:
                # Check if realtime output exists with this device_id
                realtime_file = os.path.join(dir_path, f"A*_D{device_id.zfill(3)}_realtime_output.json")
                if glob.glob(realtime_file):
                    player_dirs = [dir_path]
                    break
        
        if not player_dirs:
            return None
        
        player_dir = player_dirs[0]
        
        # Find realtime output file for this device
        realtime_pattern = os.path.join(player_dir, f"A*_D{device_id.zfill(3)}_realtime_output.json")
        realtime_files = glob.glob(realtime_pattern)
        
        if not realtime_files:
            return None
        
        # Get the most recent file if multiple exist
        realtime_file = max(realtime_files, key=os.path.getmtime)
        
        try:
            with open(realtime_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading prediction file: {e}")
            return None
    
    def get_all_active_predictions(self) -> List[Dict]:
        """
        Get latest predictions for all active devices.
        
        Returns:
            List of prediction dictionaries
        """
        try:
            output_dir = self._resolve_output_dir()
            
            # Check if directory exists
            if not os.path.exists(output_dir):
                print(f"Prediction outputs directory does not exist: {output_dir}")
                return []
            
            predictions = []
            
            # Find all player directories
            player_pattern = os.path.join(output_dir, "A*")
            player_dirs = glob.glob(player_pattern)
            
            if not player_dirs:
                # No player directories found - return empty list
                return []
            
            seen_devices = set()
            
            for player_dir in player_dirs:
                try:
                    # Find all realtime output files
                    realtime_pattern = os.path.join(player_dir, "*_realtime_output.json")
                    realtime_files = glob.glob(realtime_pattern)
                    
                    for realtime_file in realtime_files:
                        try:
                            # Extract device_id from filename
                            # Pattern: A{athlete_id}_D{device_id}_realtime_output.json
                            filename = os.path.basename(realtime_file)
                            match = re.search(r'_D(\d+)_realtime_output\.json', filename)
                            
                            if match:
                                device_id = match.group(1)
                                
                                # Only add once per device
                                if device_id not in seen_devices:
                                    seen_devices.add(device_id)
                                    
                                    try:
                                        with open(realtime_file, 'r', encoding='utf-8') as f:
                                            prediction = json.load(f)
                                            predictions.append(prediction)
                                    except json.JSONDecodeError as e:
                                        print(f"Invalid JSON in prediction file {realtime_file}: {e}")
                                        continue
                                    except Exception as e:
                                        print(f"Error reading prediction file {realtime_file}: {e}")
                                        continue
                        except Exception as e:
                            print(f"Error processing file {realtime_file}: {e}")
                            continue
                except Exception as e:
                    print(f"Error processing player directory {player_dir}: {e}")
                    continue
            
            return predictions
        except Exception as e:
            print(f"Error in get_all_active_predictions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_session_summaries(self, device_id: str, limit: int = 10) -> List[Dict]:
        """
        Get session summaries for a device.
        
        Args:
            device_id: Device ID
            limit: Maximum number of summaries to return
            
        Returns:
            List of session summary dictionaries
        """
        output_dir = self._resolve_output_dir()
        summaries = []
        
        # Find player folder for this device
        pattern = os.path.join(output_dir, f"A*_Device_*{device_id.zfill(3)}*")
        player_dirs = glob.glob(pattern)
        
        if not player_dirs:
            # Try alternative pattern
            pattern = os.path.join(output_dir, "A*")
            all_dirs = glob.glob(pattern)
            for dir_path in all_dirs:
                summary_pattern = os.path.join(dir_path, f"*_D{device_id.zfill(3)}_*_session_summary_*.json")
                if glob.glob(summary_pattern):
                    player_dirs = [dir_path]
                    break
        
        if not player_dirs:
            return summaries
        
        player_dir = player_dirs[0]
        
        # Find session summary files
        summary_pattern = os.path.join(player_dir, f"*_D{device_id.zfill(3)}_*_session_summary_*.json")
        summary_files = glob.glob(summary_pattern)
        
        # Sort by modification time (newest first)
        summary_files.sort(key=os.path.getmtime, reverse=True)
        
        for summary_file in summary_files[:limit]:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    summary['file_path'] = summary_file
                    summary['timestamp'] = datetime.fromtimestamp(os.path.getmtime(summary_file)).isoformat()
                    summaries.append(summary)
            except Exception as e:
                print(f"Error reading summary file {summary_file}: {e}")
                continue
        
        return summaries
    
    def get_active_players(self) -> List[Dict]:
        """
        Get list of all active players with their latest status.
        
        Returns:
            List of player information dictionaries
        """
        try:
            predictions = self.get_all_active_predictions()
            players = []
            
            for pred in predictions:
                try:
                    player_info = {
                        "player_id": pred.get("athlete_id"),
                        "device_id": pred.get("device_id"),
                        "status": "active",
                        "last_update": pred.get("timestamp"),
                        "profile": pred.get("athlete_profile"),
                        "latest_metrics": {
                            "heart_rate": pred.get("heart_rate"),
                            "stress_percent": pred.get("stress_percent"),
                            "total_trimp": pred.get("total_trimp"),
                            "vo2_max": pred.get("vo2_max"),
                            "g_impact_count": pred.get("g_impact_count")
                        }
                    }
                    players.append(player_info)
                except Exception as e:
                    # Log error but continue processing other predictions
                    print(f"Error processing prediction for player: {e}")
                    continue
            
            return players
        except Exception as e:
            # Return empty list on error instead of raising exception
            print(f"Error in get_active_players: {e}")
            import traceback
            traceback.print_exc()
            return []

