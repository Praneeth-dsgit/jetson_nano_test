"""
Configuration management for the athlete monitoring system.

This package contains configuration utilities:
- jetson_orin_32gb_config.yaml: Main system configuration
"""

import yaml
import os

def load_config(config_file='jetson_orin_32gb_config.yaml'):
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_config_value(key, default=None):
    """Get a specific configuration value."""
    config = load_config()
    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value

__version__ = "1.0.0"

__all__ = [
    'load_config',
    'get_config_value'
]