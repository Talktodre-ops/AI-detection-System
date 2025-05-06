import pytest

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "api: mark tests that test API functionality")
    config.addinivalue_line("markers", "data: mark tests that test data processing")
    config.addinivalue_line("markers", "model: mark tests that test model functionality")
    config.addinivalue_line("markers", "frontend: mark tests that test frontend functionality")
