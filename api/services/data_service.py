"""
Service for accessing training data and other data files.
"""

import os
import glob
from typing import List, Optional, Dict
import json


class DataService:
    """Service to access training data and other data files."""
    
    def __init__(self):
        """Initialize data service."""
        self.base_dir = self._resolve_base_dir()
    
    def _resolve_base_dir(self) -> str:
        """Resolve the data directory path."""
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        alt_paths = [
            os.path.join(current_dir, 'data'),
            os.path.join('data'),
            os.path.join('..', 'data')
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return os.path.abspath(os.path.join(current_dir, 'data'))
    
    def get_training_data_files(self, player_id: str) -> List[str]:
        """
        Get list of training data files for a player.
        
        Args:
            player_id: Player ID (e.g., "1", "2")
            
        Returns:
            List of file paths
        """
        training_dir = os.path.join(self.base_dir, 'athlete_training_data', f'player_{player_id}')
        
        if not os.path.exists(training_dir):
            return []
        
        files = glob.glob(os.path.join(training_dir, '*.csv'))
        return sorted(files)
    
    def get_game_data_files(self, player_id: str) -> List[str]:
        """
        Get list of game data files for a player.
        
        Args:
            player_id: Player ID
            
        Returns:
            List of file paths
        """
        game_dir = os.path.join(self.base_dir, 'athlete_game_data', f'player_{player_id}')
        
        if not os.path.exists(game_dir):
            return []
        
        files = glob.glob(os.path.join(game_dir, '*.csv'))
        return sorted(files)
    
    def get_data_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get information about a data file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            File information dictionary
        """
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime
        }

