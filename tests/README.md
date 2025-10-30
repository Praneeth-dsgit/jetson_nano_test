# Test Suite for Jetson Nano ML System

This directory contains unit tests for the core functionality of the Jetson Nano ML prediction system.

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared pytest fixtures
├── test_health_metrics.py   # Health metrics calculations
├── test_sensor_processing.py # Sensor data processing
└── test_utils.py            # Utility functions
```

## Running Tests

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_health_metrics.py
```

### Run Specific Test Class
```bash
pytest tests/test_health_metrics.py::TestCalculateRMSSD
```

### Run Specific Test Function
```bash
pytest tests/test_health_metrics.py::TestCalculateRMSSD::test_normal_hr_values
```

### Run with Coverage
```bash
pytest --cov=core --cov-report=html
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with More Details
```bash
pytest -vv
```

## Test Coverage

### Health Metrics (`test_health_metrics.py`)
- ✅ `calculate_rmssd` - HRV calculation
- ✅ `calculate_stress` - Stress level calculation
- ✅ `estimate_vo2_max` - VO2 max estimation
- ✅ `calculate_trimp` - TRIMP calculation
- ✅ `training_energy_expenditure` - Energy expenditure
- ✅ `get_trimp_zone` - TRIMP zone classification
- ✅ `get_recovery_recommendations` - Recovery recommendations
- ✅ `get_training_recommendations` - Training recommendations

### Sensor Processing (`test_sensor_processing.py`)
- ✅ `parse_sensor_payload` - MQTT payload parsing
- ✅ `butter_bandpass_filter` - Signal filtering
- ✅ `compute_window_features` - Feature engineering
- ✅ `_safe_prepare_input` - Input preparation

### Utilities (`test_utils.py`)
- ✅ `_init_device_context` - Device context initialization
- ✅ `get_memory_status` - Memory monitoring
- ✅ `get_device` - Device detection

## Writing New Tests

### Basic Test Structure
```python
def test_function_name():
    """Test description."""
    result = function_to_test(input_data)
    assert result == expected_value
```

### Using Fixtures
```python
def test_with_fixture(sample_hr_values):
    """Test using a fixture."""
    result = calculate_rmssd(sample_hr_values)
    assert result > 0
```

### Testing Exceptions
```python
def test_invalid_input():
    """Test that invalid input raises exception."""
    with pytest.raises(ValueError):
        calculate_trimp(150, 60, 200, 30, "invalid")
```

## Best Practices

1. **Test Names**: Use descriptive names starting with `test_`
2. **One Assertion**: Each test should verify one thing
3. **Edge Cases**: Test boundary conditions and error cases
4. **Fixtures**: Use fixtures for common test data
5. **Independence**: Tests should not depend on each other
6. **Fast**: Tests should run quickly

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: pytest tests/
```

## Notes

- Tests use `pytest` framework
- Some tests may require mocked dependencies (MQTT, database, etc.)
- Tests are designed to run without external dependencies where possible
- For integration tests, consider using `pytest-mock` for mocking

