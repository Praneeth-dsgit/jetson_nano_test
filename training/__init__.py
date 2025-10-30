"""
Training and model management modules.

This package contains machine learning training and model versioning:
- sup_ml_rf_training: ML training pipeline with validation
- model_version_manager: Model versioning, backup, and rollback
"""

from .sup_ml_rf_training import *
from .model_version_manager import ModelVersionManager

__version__ = "1.0.0"

__all__ = [
    'ModelVersionManager'
]