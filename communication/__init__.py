"""
Communication modules for MQTT and data publishing.

This package contains communication functionality:
- mqtt_message_queue: Reliable MQTT messaging with persistence
- publisher: Data publisher for sensor data simulation
"""

from .mqtt_message_queue import MQTTMessageQueue
from .publisher import *

__version__ = "1.0.0"

__all__ = [
    'MQTTMessageQueue'
]