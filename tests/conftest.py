#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures.

This file contains fixtures that can be used across all test files.
"""

import pytest
import numpy as np
from collections import deque


@pytest.fixture
def sample_hr_values():
    """Sample heart rate values for testing."""
    return [60, 62, 65, 67, 64, 66, 68, 70, 72, 71]


@pytest.fixture
def sample_hr_values_short():
    """Short sample HR values (insufficient for RMSSD)."""
    return [60, 62, 65]


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data dictionary."""
    return {
        "acc": {"x": 1.0, "y": 2.0, "z": 3.0},
        "gyro": {"x": 10.0, "y": 20.0, "z": 30.0},
        "magno": {"x": 0.5, "y": 0.6, "z": 0.7}
    }


@pytest.fixture
def sample_window_data():
    """Sample window data for feature engineering."""
    return {
        "ax": deque([1.0] * 30, maxlen=30),
        "ay": deque([2.0] * 30, maxlen=30),
        "az": deque([3.0] * 30, maxlen=30),
        "gx": deque([10.0] * 30, maxlen=30),
        "gy": deque([20.0] * 30, maxlen=30),
        "gz": deque([30.0] * 30, maxlen=30)
    }


@pytest.fixture
def sample_mqtt_payload():
    """Sample MQTT payload (JSON format)."""
    import json
    payload_data = {
        "device_id": "001",
        "athlete_id": 1,
        "acc_x": 1.0,
        "acc_y": 2.0,
        "acc_z": 3.0,
        "gyro_x": 10.0,
        "gyro_y": 20.0,
        "gyro_z": 30.0,
        "mag_x": 0.5,
        "mag_y": 0.6,
        "mag_z": 0.7,
        "mode": "game"
    }
    return json.dumps(payload_data).encode()


@pytest.fixture
def sample_filter_signal():
    """Sample signal for filtering tests."""
    t = np.linspace(0, 1, 100)
    return np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

