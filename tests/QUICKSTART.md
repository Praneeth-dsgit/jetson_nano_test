# Quick Start Guide for Running Tests

## Prerequisites

```bash
# Install test dependencies
pip install -r requirements.txt
```

## Basic Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_health_metrics.py

# Run with verbose output
pytest -v

# Run with extra verbose output
pytest -vv

# Run specific test
pytest tests/test_health_metrics.py::TestCalculateRMSSD::test_normal_hr_values

# Run tests matching a pattern
pytest -k "test_calculate"

# Run tests and show coverage
pytest --cov=core --cov-report=html

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

## Expected Output

When tests pass, you'll see:
```
========================= test session starts =========================
collected 50 items

tests/test_health_metrics.py::TestCalculateRMSSD::test_normal_hr_values PASSED
tests/test_health_metrics.py::TestCalculateRMSSD::test_insufficient_data PASSED
...
========================= 50 passed in 2.34s =========================
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd /path/to/jetson_nano_test-1
pytest
```

### Missing Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

### Module Not Found
Make sure your PYTHONPATH includes the project root, or run:
```bash
PYTHONPATH=. pytest
```

