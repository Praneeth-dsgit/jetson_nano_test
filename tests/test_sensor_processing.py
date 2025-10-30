#!/usr/bin/env python3
"""
Test cases for sensor data processing functions.

Tests include:
- parse_sensor_payload (MQTT payload parsing)
- butter_bandpass_filter (signal filtering)
- compute_window_features (feature engineering)
"""

import pytest
import numpy as np
import json
from collections import deque
from core.main import (
    parse_sensor_payload,
    butter_bandpass_filter,
    compute_window_features,
    _safe_prepare_input,
    _infer_input_dim_for_model
)


class TestParseSensorPayload:
    """Test cases for parse_sensor_payload function."""
    
    def test_valid_json_payload(self):
        """Test parsing valid JSON payload."""
        payload_data = {
            "acc_x": 1.0,
            "acc_y": 2.0,
            "acc_z": 3.0,
            "gyro_x": 10.0,
            "device_id": "001"
        }
        payload = json.dumps(payload_data).encode()
        result = parse_sensor_payload(payload)
        assert result == payload_data
    
    def test_simple_format_payload(self):
        """Test parsing simple format (x/y/z)."""
        payload = b"1.5/2.5/3.5"
        result = parse_sensor_payload(payload)
        assert result == {"x": 1.5, "y": 2.5, "z": 3.5}
    
    def test_simple_format_with_spaces(self):
        """Test simple format with spaces."""
        payload = b" 1.5 / 2.5 / 3.5 "
        result = parse_sensor_payload(payload)
        assert result == {"x": 1.5, "y": 2.5, "z": 3.5}
    
    def test_invalid_json_fallback(self):
        """Test that invalid JSON falls back to simple format."""
        payload = b"1.0/2.0/3.0"
        result = parse_sensor_payload(payload)
        assert result == {"x": 1.0, "y": 2.0, "z": 3.0}
    
    def test_invalid_payload_returns_none(self):
        """Test that completely invalid payload returns None."""
        payload = b"invalid_data_format"
        result = parse_sensor_payload(payload)
        assert result is None
    
    def test_empty_payload(self):
        """Test empty payload."""
        payload = b""
        result = parse_sensor_payload(payload)
        # Should return None or handle gracefully
        assert result is None or isinstance(result, dict)
    
    def test_malformed_json(self):
        """Test malformed JSON."""
        payload = b"{invalid json}"
        result = parse_sensor_payload(payload)
        # Should try fallback format or return None
        assert result is None or isinstance(result, dict)
    
    def test_simple_format_wrong_parts(self):
        """Test simple format with wrong number of parts."""
        payload = b"1.0/2.0"  # Only 2 parts, needs 3
        result = parse_sensor_payload(payload)
        assert result is None


class TestButterBandpassFilter:
    """Test cases for butter_bandpass_filter function."""
    
    def test_normal_filtering(self):
        """Test normal filtering operation."""
        # Create test signal (sine wave)
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        
        filtered = butter_bandpass_filter(signal, low_cutoff=0.3, high_cutoff=4.5, fs=10.0)
        assert len(filtered) == len(signal)
        assert isinstance(filtered, np.ndarray)
    
    def test_list_input(self):
        """Test that list input works."""
        signal = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        filtered = butter_bandpass_filter(signal, fs=10.0)
        assert len(filtered) == len(signal)
    
    def test_different_frequencies(self):
        """Test with different frequency parameters."""
        signal = np.random.randn(100)
        filtered1 = butter_bandpass_filter(signal, low_cutoff=0.5, high_cutoff=2.0, fs=10.0)
        filtered2 = butter_bandpass_filter(signal, low_cutoff=1.0, high_cutoff=3.0, fs=10.0)
        assert len(filtered1) == len(filtered2)
    
    def test_custom_order(self):
        """Test with different filter orders."""
        signal = np.random.randn(100)
        filtered = butter_bandpass_filter(signal, order=4, fs=10.0)
        assert len(filtered) == len(signal)
    
    def test_output_shape(self):
        """Test that output shape matches input."""
        signal = np.random.randn(50)
        filtered = butter_bandpass_filter(signal, fs=10.0)
        assert filtered.shape == signal.shape


class TestComputeWindowFeatures:
    """Test cases for compute_window_features function."""
    
    def test_sufficient_data(self):
        """Test with sufficient window data."""
        # Create test data with enough samples (need WINDOW_SAMPLES = 30)
        ax = deque([1.0] * 30, maxlen=30)
        ay = deque([2.0] * 30, maxlen=30)
        az = deque([3.0] * 30, maxlen=30)
        gx = deque([10.0] * 30, maxlen=30)
        gy = deque([20.0] * 30, maxlen=30)
        gz = deque([30.0] * 30, maxlen=30)
        
        result = compute_window_features(ax, ay, az, gx, gy, gz)
        assert result is not None
        assert isinstance(result, dict)
        assert "acc_x_mean" in result
        assert "gyro_z_mean" in result
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        ax = deque([1.0] * 5, maxlen=30)
        ay = deque([2.0] * 5, maxlen=30)
        az = deque([3.0] * 5, maxlen=30)
        gx = deque([10.0] * 5, maxlen=30)
        gy = deque([20.0] * 5, maxlen=30)
        gz = deque([30.0] * 5, maxlen=30)
        
        result = compute_window_features(ax, ay, az, gx, gy, gz)
        assert result is None
    
    def test_feature_completeness(self):
        """Test that all expected features are present."""
        ax = deque([1.0] * 30, maxlen=30)
        ay = deque([2.0] * 30, maxlen=30)
        az = deque([3.0] * 30, maxlen=30)
        gx = deque([10.0] * 30, maxlen=30)
        gy = deque([20.0] * 30, maxlen=30)
        gz = deque([30.0] * 30, maxlen=30)
        
        result = compute_window_features(ax, ay, az, gx, gy, gz)
        assert result is not None
        
        # Check for key feature groups
        expected_features = [
            "acc_x_mean", "acc_x_std", "acc_x_min", "acc_x_max", "acc_x_range",
            "gyro_x_mean", "gyro_x_std",
            "resultant_mean", "resultant_std",
            "mean_abs_jerk",
            "spectral_energy_0p5_5hz",
            "peak_frequency_hz",
            "player_load_index"
        ]
        
        for feature in expected_features:
            assert feature in result, f"Missing feature: {feature}"
            assert isinstance(result[feature], (int, float))
    
    def test_exception_handling(self):
        """Test that exceptions return None gracefully."""
        # Create invalid data that might cause issues
        ax = deque([float('inf')] * 30, maxlen=30)
        ay = deque([2.0] * 30, maxlen=30)
        az = deque([3.0] * 30, maxlen=30)
        gx = deque([10.0] * 30, maxlen=30)
        gy = deque([20.0] * 30, maxlen=30)
        gz = deque([30.0] * 30, maxlen=30)
        
        # Should handle gracefully (returns None or dict with defaults)
        result = compute_window_features(ax, ay, az, gx, gy, gz)
        # Result might be None or a dict with default values
        assert result is None or isinstance(result, dict)


class TestSafePrepareInput:
    """Test cases for _safe_prepare_input function."""
    
    def test_padding(self):
        """Test that short arrays are padded."""
        features = [1.0, 2.0, 3.0]
        result = _safe_prepare_input(features, required_dim=10)
        assert len(result) == 10
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] == 3.0
        assert result[3] == 0.0  # Padded zeros
    
    def test_truncation(self):
        """Test that long arrays are truncated."""
        features = list(range(20))
        result = _safe_prepare_input(features, required_dim=10)
        assert len(result) == 10
        assert result[0] == 0.0
        assert result[9] == 9.0
    
    def test_exact_length(self):
        """Test with exact required length."""
        features = [1.0, 2.0, 3.0]
        result = _safe_prepare_input(features, required_dim=3)
        assert len(result) == 3
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _safe_prepare_input(features, required_dim=5)
        assert len(result) == 5
        assert isinstance(result, np.ndarray)
    
    def test_zero_length_input(self):
        """Test with zero-length input."""
        features = []
        result = _safe_prepare_input(features, required_dim=5)
        assert len(result) == 5
        assert np.all(result == 0.0)

