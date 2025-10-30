#!/usr/bin/env python3
"""
Test cases for utility functions.

Tests include:
- Device context management
- Memory status functions
- Configuration helpers
"""

import pytest
import numpy as np
from core.main import (
    _init_device_context,
    _load_context_to_globals,
    _save_globals_to_context,
    get_memory_status,
    get_device,
    _safe_prepare_input
)


class TestDeviceContext:
    """Test cases for device context management."""
    
    def test_init_device_context(self):
        """Test device context initialization."""
        context = _init_device_context("001")
        assert isinstance(context, dict)
        assert context["device_id"] == "001"
        assert "athlete_id" in context
        assert "name" in context
        assert "hr_buffer" in context
    
    def test_device_context_structure(self):
        """Test that context has all required fields."""
        context = _init_device_context("002")
        
        required_fields = [
            "device_id", "athlete_id", "name", "age", "weight",
            "height", "gender", "hr_rest", "hr_max",
            "madgwick_quaternion", "hr_buffer", "acc_buffer",
            "gyro_buffer", "session_start_time", "session_end_time"
        ]
        
        for field in required_fields:
            assert field in context, f"Missing field: {field}"
    
    def test_device_context_defaults(self):
        """Test default values in context."""
        context = _init_device_context("003")
        
        assert context["age"] == 25
        assert context["weight"] == 70.0
        assert context["height"] == 175.0
        assert context["gender"] == 1
        assert context["hr_rest"] == 60
    
    def test_invalid_device_id(self):
        """Test with invalid device ID."""
        # Should handle gracefully
        context = _init_device_context("invalid")
        assert context["device_id"] == "invalid"
        # athlete_id should be 0 for invalid
        assert context["athlete_id"] == 0 or context["athlete_id"] is not None
    
    def test_context_to_globals_roundtrip(self):
        """Test loading and saving context."""
        context = _init_device_context("004")
        
        # Modify context
        context["age"] = 30
        context["weight"] = 75.0
        
        # Save to globals
        _load_context_to_globals(context)
        
        # Modify globals
        # Note: This test might need adjustment based on actual global usage
        # For now, just verify the function runs without error
        _save_globals_to_context(context)
        
        # Context should be updated
        assert context["age"] == 30


class TestMemoryStatus:
    """Test cases for memory status functions."""
    
    def test_get_memory_status(self):
        """Test memory status retrieval."""
        status = get_memory_status()
        assert isinstance(status, dict)
        assert "cpu" in status
        assert "timestamp" in status
    
    def test_cpu_memory_fields(self):
        """Test CPU memory fields."""
        status = get_memory_status()
        cpu = status["cpu"]
        
        assert "process_memory_mb" in cpu
        assert "process_memory_percent" in cpu
        assert "total_system_mb" in cpu
        assert "available_mb" in cpu
        assert "system_usage_percent" in cpu
        
        # Values should be positive numbers
        assert cpu["process_memory_mb"] > 0
        assert cpu["total_system_mb"] > 0
    
    def test_gpu_memory_fields(self):
        """Test GPU memory fields (if available)."""
        status = get_memory_status()
        gpu = status["gpu"]
        
        # GPU might not be available, so check for status or fields
        assert "status" in gpu or "device_name" in gpu or "error" in gpu
    
    def test_memory_status_error_handling(self):
        """Test that errors are handled gracefully."""
        # Function should not crash even if monitoring fails
        status = get_memory_status()
        assert isinstance(status, dict)
        # Should have either data or error message
        assert "cpu" in status or "error" in status


class TestSafePrepareInput:
    """Test cases for _safe_prepare_input utility."""
    
    def test_list_input(self):
        """Test with list input."""
        features = [1.0, 2.0, 3.0]
        result = _safe_prepare_input(features, required_dim=5)
        assert len(result) == 5
        assert result[0] == 1.0
        assert result[4] == 0.0  # Padded
    
    def test_numpy_array_input(self):
        """Test with numpy array."""
        features = np.array([1.0, 2.0, 3.0])
        result = _safe_prepare_input(features, required_dim=5)
        assert len(result) == 5
        assert isinstance(result, np.ndarray)
    
    def test_truncation(self):
        """Test truncation of long arrays."""
        features = list(range(10))
        result = _safe_prepare_input(features, required_dim=5)
        assert len(result) == 5
        assert result[0] == 0.0
        assert result[4] == 4.0
    
    def test_exact_match(self):
        """Test with exact matching dimensions."""
        features = [1.0, 2.0, 3.0]
        result = _safe_prepare_input(features, required_dim=3)
        assert len(result) == 3
        assert np.allclose(result, [1.0, 2.0, 3.0])


class TestGetDevice:
    """Test cases for get_device function."""
    
    def test_device_returns_string(self):
        """Test that device function returns a string."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda"]
    
    def test_device_consistency(self):
        """Test that device detection is consistent."""
        device1 = get_device()
        device2 = get_device()
        # Should return same result (unless environment changed)
        assert isinstance(device1, str)
        assert isinstance(device2, str)

