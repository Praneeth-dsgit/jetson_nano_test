#!/usr/bin/env python3
"""
Model Version Manager for Athlete Monitoring System

This module provides comprehensive model versioning including:
- Automatic model backup and versioning
- Model rollback capabilities
- Model performance tracking
- Version comparison and validation
- Model deployment management
"""

import os
import json
import shutil
import sqlite3
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import joblib
import numpy as np
from pathlib import Path

class ModelStatus(Enum):
    ACTIVE = "active"
    BACKUP = "backup"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    version_id: str
    player_id: str
    model_type: str  # 'pkl' or 'hb'
    file_path: str
    backup_path: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    model_hash: str
    status: ModelStatus
    notes: Optional[str] = None
    parent_version: Optional[str] = None

class ModelVersionManager:
    """
    Comprehensive model version management system.
    
    Features:
    - Automatic model backup and versioning
    - Model rollback capabilities
    - Performance tracking and comparison
    - Model validation and integrity checks
    - Deployment management
    """
    
    def __init__(self, 
                 backup_root: str = "model_backups",
                 db_path: str = "model_versions.db",
                 max_versions_per_model: int = 10,
                 enable_logging: bool = True):
        """
        Initialize the model version manager.
        
        Args:
            backup_root: Root directory for model backups
            db_path: Path to SQLite database for version tracking
            max_versions_per_model: Maximum number of versions to keep per model
            enable_logging: Whether to enable detailed logging
        """
        self.backup_root = backup_root
        self.db_path = db_path
        self.max_versions_per_model = max_versions_per_model
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Create backup directory
        os.makedirs(backup_root, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self.stats = {
            "total_versions": 0,
            "active_models": 0,
            "backup_models": 0,
            "rollbacks_performed": 0,
            "models_deployed": 0
        }
    
    def _init_database(self):
        """Initialize SQLite database for version tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create versions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    backup_path TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    model_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    parent_version TEXT,
                    UNIQUE(player_id, model_type, version_id)
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_type ON model_versions(player_id, model_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON model_versions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON model_versions(created_at)')
            
            conn.commit()
            conn.close()
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Model version database initialized: {self.db_path}")
                
        except Exception as e:
            error_msg = f"Failed to initialize model version database: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _calculate_model_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of model file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error calculating model hash: {e}")
            return "unknown"
    
    def _generate_version_id(self, player_id: str, model_type: str) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{player_id}_{model_type}_{timestamp}"
    
    def backup_model(self, 
                    player_id: str, 
                    model_path: str, 
                    model_type: str = "pkl",
                    performance_metrics: Optional[Dict[str, float]] = None,
                    notes: Optional[str] = None) -> str:
        """
        Backup a model with versioning.
        
        Args:
            player_id: Player ID
            model_path: Path to the model file
            model_type: Type of model ('pkl' or 'hb')
            performance_metrics: Model performance metrics
            notes: Optional notes about this version
            
        Returns:
            Version ID of the backup
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Generate version ID and backup path
            version_id = self._generate_version_id(player_id, model_type)
            backup_dir = os.path.join(self.backup_root, player_id, model_type)
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_filename = f"{version_id}.{model_type}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy model file to backup location
            shutil.copy2(model_path, backup_path)
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(model_path)
            
            # Get current active version as parent
            parent_version = self.get_active_version_id(player_id, model_type)
            
            # Create version record
            version = ModelVersion(
                version_id=version_id,
                player_id=player_id,
                model_type=model_type,
                file_path=model_path,
                backup_path=backup_path,
                created_at=datetime.now(),
                performance_metrics=performance_metrics or {},
                model_hash=model_hash,
                status=ModelStatus.BACKUP,
                notes=notes,
                parent_version=parent_version
            )
            
            # Save to database
            self._save_version_to_db(version)
            
            # Clean up old versions if needed
            self._cleanup_old_versions(player_id, model_type)
            
            self.stats["total_versions"] += 1
            self.stats["backup_models"] += 1
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Model backed up: {version_id} -> {backup_path}")
            
            return version_id
            
        except Exception as e:
            error_msg = f"Failed to backup model {player_id}/{model_type}: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def deploy_model(self, 
                    player_id: str, 
                    version_id: str, 
                    target_path: str,
                    model_type: str = "pkl") -> bool:
        """
        Deploy a specific model version to the target path.
        
        Args:
            player_id: Player ID
            version_id: Version ID to deploy
            target_path: Target path for deployment
            model_type: Type of model ('pkl' or 'hb')
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Get version information
            version = self.get_version_info(version_id)
            if not version:
                raise ValueError(f"Version not found: {version_id}")
            
            if version.player_id != player_id or version.model_type != model_type:
                raise ValueError(f"Version mismatch: expected {player_id}/{model_type}, got {version.player_id}/{version.model_type}")
            
            # Create target directory if needed
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Copy backup to target location
            shutil.copy2(version.backup_path, target_path)
            
            # Update version status to active
            self._update_version_status(version_id, ModelStatus.ACTIVE)
            
            # Mark previous active version as backup
            previous_active = self.get_active_version_id(player_id, model_type)
            if previous_active and previous_active != version_id:
                self._update_version_status(previous_active, ModelStatus.BACKUP)
            
            self.stats["models_deployed"] += 1
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Model deployed: {version_id} -> {target_path}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to deploy model {version_id}: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            return False
    
    def rollback_model(self, 
                      player_id: str, 
                      model_type: str = "pkl",
                      target_path: str = None) -> Optional[str]:
        """
        Rollback to the previous model version.
        
        Args:
            player_id: Player ID
            model_type: Type of model ('pkl' or 'hb')
            target_path: Target path for rollback (if None, uses original path)
            
        Returns:
            Version ID of the rolled back model, or None if rollback failed
        """
        try:
            # Get current active version
            current_version = self.get_active_version_id(player_id, model_type)
            if not current_version:
                raise ValueError(f"No active version found for {player_id}/{model_type}")
            
            # Get parent version
            current_info = self.get_version_info(current_version)
            if not current_info or not current_info.parent_version:
                raise ValueError(f"No parent version found for {current_version}")
            
            parent_version = current_info.parent_version
            
            # Deploy parent version
            if target_path is None:
                target_path = current_info.file_path
            
            success = self.deploy_model(player_id, parent_version, target_path, model_type)
            
            if success:
                # Mark current version as deprecated
                self._update_version_status(current_version, ModelStatus.DEPRECATED)
                
                self.stats["rollbacks_performed"] += 1
                
                if self.enable_logging and self.logger:
                    self.logger.info(f"Model rolled back: {current_version} -> {parent_version}")
                
                return parent_version
            else:
                return None
                
        except Exception as e:
            error_msg = f"Failed to rollback model {player_id}/{model_type}: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            return None
    
    def get_active_version_id(self, player_id: str, model_type: str) -> Optional[str]:
        """Get the active version ID for a player and model type."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT version_id FROM model_versions 
                WHERE player_id = ? AND model_type = ? AND status = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (player_id, model_type, ModelStatus.ACTIVE.value))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error getting active version: {e}")
            return None
    
    def get_version_info(self, version_id: str) -> Optional[ModelVersion]:
        """Get detailed information about a specific version."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT version_id, player_id, model_type, file_path, backup_path,
                       created_at, performance_metrics, model_hash, status, notes, parent_version
                FROM model_versions WHERE version_id = ?
            ''', (version_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return ModelVersion(
                    version_id=result[0],
                    player_id=result[1],
                    model_type=result[2],
                    file_path=result[3],
                    backup_path=result[4],
                    created_at=datetime.fromisoformat(result[5]),
                    performance_metrics=json.loads(result[6]),
                    model_hash=result[7],
                    status=ModelStatus(result[8]),
                    notes=result[9],
                    parent_version=result[10]
                )
            return None
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error getting version info: {e}")
            return None
    
    def list_versions(self, player_id: str, model_type: str = None) -> List[ModelVersion]:
        """List all versions for a player, optionally filtered by model type."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if model_type:
                cursor.execute('''
                    SELECT version_id, player_id, model_type, file_path, backup_path,
                           created_at, performance_metrics, model_hash, status, notes, parent_version
                    FROM model_versions 
                    WHERE player_id = ? AND model_type = ?
                    ORDER BY created_at DESC
                ''', (player_id, model_type))
            else:
                cursor.execute('''
                    SELECT version_id, player_id, model_type, file_path, backup_path,
                           created_at, performance_metrics, model_hash, status, notes, parent_version
                    FROM model_versions 
                    WHERE player_id = ?
                    ORDER BY created_at DESC
                ''', (player_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            versions = []
            for result in results:
                versions.append(ModelVersion(
                    version_id=result[0],
                    player_id=result[1],
                    model_type=result[2],
                    file_path=result[3],
                    backup_path=result[4],
                    created_at=datetime.fromisoformat(result[5]),
                    performance_metrics=json.loads(result[6]),
                    model_hash=result[7],
                    status=ModelStatus(result[8]),
                    notes=result[9],
                    parent_version=result[10]
                ))
            
            return versions
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error listing versions: {e}")
            return []
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        try:
            version1 = self.get_version_info(version_id1)
            version2 = self.get_version_info(version_id2)
            
            if not version1 or not version2:
                return {"error": "One or both versions not found"}
            
            comparison = {
                "version1": {
                    "id": version1.version_id,
                    "created_at": version1.created_at.isoformat(),
                    "performance": version1.performance_metrics,
                    "status": version1.status.value,
                    "notes": version1.notes
                },
                "version2": {
                    "id": version2.version_id,
                    "created_at": version2.created_at.isoformat(),
                    "performance": version2.performance_metrics,
                    "status": version2.status.value,
                    "notes": version2.notes
                },
                "comparison": {
                    "same_hash": version1.model_hash == version2.model_hash,
                    "time_diff_hours": abs((version1.created_at - version2.created_at).total_seconds() / 3600),
                    "performance_diff": {}
                }
            }
            
            # Compare performance metrics
            all_metrics = set(version1.performance_metrics.keys()) | set(version2.performance_metrics.keys())
            for metric in all_metrics:
                val1 = version1.performance_metrics.get(metric, 0)
                val2 = version2.performance_metrics.get(metric, 0)
                comparison["comparison"]["performance_diff"][metric] = val2 - val1
            
            return comparison
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}
    
    def validate_model_integrity(self, version_id: str) -> Dict[str, Any]:
        """Validate model file integrity."""
        try:
            version = self.get_version_info(version_id)
            if not version:
                return {"valid": False, "error": "Version not found"}
            
            # Check if backup file exists
            if not os.path.exists(version.backup_path):
                return {"valid": False, "error": "Backup file not found"}
            
            # Recalculate hash and compare
            current_hash = self._calculate_model_hash(version.backup_path)
            hash_match = current_hash == version.model_hash
            
            # Try to load the model
            load_success = False
            load_error = None
            try:
                if version.model_type == "pkl":
                    joblib.load(version.backup_path)
                elif version.model_type == "hb":
                    # For Hummingbird models, we'd need to use the appropriate loader
                    # This is a placeholder - actual implementation would depend on the HB format
                    pass
                load_success = True
            except Exception as e:
                load_error = str(e)
            
            return {
                "valid": hash_match and load_success,
                "hash_match": hash_match,
                "load_success": load_success,
                "load_error": load_error,
                "file_size": os.path.getsize(version.backup_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(version.backup_path)).isoformat()
            }
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error validating model integrity: {e}")
            return {"valid": False, "error": str(e)}
    
    def _save_version_to_db(self, version: ModelVersion):
        """Save version information to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_versions 
                (version_id, player_id, model_type, file_path, backup_path, created_at,
                 performance_metrics, model_hash, status, notes, parent_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                version.version_id, version.player_id, version.model_type,
                version.file_path, version.backup_path, version.created_at.isoformat(),
                json.dumps(version.performance_metrics), version.model_hash,
                version.status.value, version.notes, version.parent_version
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error saving version to database: {e}")
            raise
    
    def _update_version_status(self, version_id: str, status: ModelStatus):
        """Update version status in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE model_versions SET status = ? WHERE version_id = ?
            ''', (status.value, version_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error updating version status: {e}")
            raise
    
    def _cleanup_old_versions(self, player_id: str, model_type: str):
        """Clean up old versions beyond the maximum limit."""
        try:
            versions = self.list_versions(player_id, model_type)
            
            if len(versions) > self.max_versions_per_model:
                # Sort by creation date (oldest first)
                versions.sort(key=lambda v: v.created_at)
                
                # Remove oldest versions beyond the limit
                versions_to_remove = versions[:-self.max_versions_per_model]
                
                for version in versions_to_remove:
                    # Only remove backup versions, not active ones
                    if version.status == ModelStatus.BACKUP:
                        try:
                            # Remove backup file
                            if os.path.exists(version.backup_path):
                                os.remove(version.backup_path)
                            
                            # Remove from database
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute('DELETE FROM model_versions WHERE version_id = ?', (version.version_id,))
                            conn.commit()
                            conn.close()
                            
                            if self.enable_logging and self.logger:
                                self.logger.info(f"Cleaned up old version: {version.version_id}")
                                
                        except Exception as e:
                            if self.enable_logging and self.logger:
                                self.logger.warning(f"Failed to cleanup version {version.version_id}: {e}")
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error cleaning up old versions: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get version management statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts by status
            cursor.execute('''
                SELECT status, COUNT(*) FROM model_versions GROUP BY status
            ''')
            status_counts = dict(cursor.fetchall())
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM model_versions')
            total_versions = cursor.fetchone()[0]
            
            # Get unique players
            cursor.execute('SELECT COUNT(DISTINCT player_id) FROM model_versions')
            unique_players = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_versions": total_versions,
                "unique_players": unique_players,
                "status_counts": status_counts,
                "stats": self.stats.copy(),
                "backup_root": self.backup_root,
                "max_versions_per_model": self.max_versions_per_model
            }
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}


# Convenience function for integration
def create_model_version_manager(backup_root: str = "model_backups",
                               db_path: str = "model_versions.db",
                               max_versions: int = 10,
                               enable_logging: bool = True) -> ModelVersionManager:
    """
    Create a model version manager with default settings.
    
    Args:
        backup_root: Root directory for model backups
        db_path: Path to SQLite database for version tracking
        max_versions: Maximum number of versions to keep per model
        enable_logging: Whether to enable detailed logging
        
    Returns:
        Configured ModelVersionManager instance
    """
    return ModelVersionManager(
        backup_root=backup_root,
        db_path=db_path,
        max_versions_per_model=max_versions,
        enable_logging=enable_logging
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create version manager
    vm = create_model_version_manager()
    
    # Test version management
    print("Testing model version manager...")
    
    # Simulate backing up a model
    test_model_path = "test_model.pkl"
    if os.path.exists(test_model_path):
        version_id = vm.backup_model(
            player_id="test_player",
            model_path=test_model_path,
            model_type="pkl",
            performance_metrics={"accuracy": 0.85, "rmse": 12.5},
            notes="Test model backup"
        )
        print(f"Backed up model: {version_id}")
        
        # List versions
        versions = vm.list_versions("test_player")
        print(f"Found {len(versions)} versions for test_player")
        
        # Get statistics
        stats = vm.get_statistics()
        print(f"Version manager stats: {stats}")
    
    print("Model version manager test completed")
