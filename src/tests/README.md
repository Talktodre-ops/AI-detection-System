# AI Detection System Tests

This directory contains tests for the AI Detection System.

## Running Tests

To run all tests:
```bash
cd src
python -m pytest tests/
```

To run specific test categories:
```bash
# Run only model tests
python -m pytest tests/ -m model

# Run only data processing tests
python -m pytest tests/ -m data

# Run only frontend tests
python -m pytest tests/ -m frontend

# Run slow tests too
python -m pytest tests/ --runslow
```

## Test Categories

- **Model Tests**: Tests for the AI detection model, including loading, prediction, and evaluation.
- **Data Processing Tests**: Tests for data loading, cleaning, and preprocessing functions.
- **Frontend Tests**: Tests for the Streamlit frontend components.
- **Slow Tests**: Tests that take a long time to run (marked with `@pytest.mark.slow`).

## Adding New Tests

1. Create a new test file in the `tests/` directory with the prefix `test_`.
2. Import the necessary modules and functions.
3. Write test functions with the prefix `test_`.
4. Use the appropriate markers to categorize your tests.

Example:
```python
import pytest

@pytest.mark.model
def test_new_model_feature():
    # Test code here
    assert True
```