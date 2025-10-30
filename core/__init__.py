"""
Core system modules for athlete monitoring.

This package contains the main processing engine and core functionality:
- main: Main prediction engine
- dynamic_model_loader: Dynamic model loading with LRU cache
- data_quality_assessor: Sensor data quality assessment
- system_health_monitor: System health monitoring and alerting
"""

# Import only the modules, not main.py to avoid circular imports
from .dynamic_model_loader import DynamicModelLoader
from .data_quality_assessor import SensorDataQualityAssessor
from .system_health_monitor import SystemHealthMonitor

__version__ = "1.0.0"
__author__ = "Athlete Monitoring System"

# Define what gets imported with "from core import *"
__all__ = [
    'DynamicModelLoader',
    'SensorDataQualityAssessor', 
    'SystemHealthMonitor'
]