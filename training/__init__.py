"""
Training and model management modules.

This package contains machine learning training and model versioning:
- sup_ml_rf_training: ML training pipeline with validation
- model_version_manager: Model versioning, backup, and rollback
"""

# Import ModelVersionManager directly to avoid circular imports
from .model_version_manager import ModelVersionManager

__version__ = "1.0.0"

__all__ = [
    'ModelVersionManager'
]

# Note: sup_ml_rf_training is not imported here to avoid circular import issues
# when the script is run directly. Import it explicitly when needed.