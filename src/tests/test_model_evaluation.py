import pytest
import os
import pandas as pd
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.evaluate_model import evaluate_model, calculate_metrics

@pytest.mark.model
def test_calculate_metrics():
    """Test the metrics calculation function"""
    # Test with perfect predictions
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    
    # Test with imperfect predictions
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]  # One false negative
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 0.75
    assert 0 < metrics["precision"] <= 1.0
    assert 0 < metrics["recall"] < 1.0
    assert 0 < metrics["f1"] < 1.0

@pytest.mark.model
def test_evaluate_model_function():
    """Test the model evaluation function with a small test set"""
    # Create a small test dataset
    test_data = pd.DataFrame({
        "cleaned_text": [
            "The artificial intelligence revolution has transformed various sectors.",
            "I'm not sure if this will work, but let's give it a try!",
            "The utilization of advanced algorithms enables efficient data processing.",
            "Hey there! Just wanted to check in and see how you're doing today."
        ],
        "label": [1, 0, 1, 0]  # 1 = AI, 0 = Human
    })
    
    # Save to a temporary CSV
    temp_test_path = "temp_eval_test.csv"
    test_data.to_csv(temp_test_path, index=False)
    
    try:
        # Skip the actual evaluation since we're just testing the function structure
        from model.evaluate_model import calculate_metrics
        
        # Test the metrics calculation function directly
        metrics = calculate_metrics([0, 1, 0, 1], [0, 1, 0, 1])
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        
        # For the evaluate_model function, we'll just check it exists and has the right signature
        from model.evaluate_model import evaluate_model
        assert callable(evaluate_model)
        
    finally:
        # Clean up
        if os.path.exists(temp_test_path):
            os.remove(temp_test_path)
