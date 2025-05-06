import pytest
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

@pytest.mark.api
def test_flag_prediction_endpoint():
    """Test the flag prediction endpoint"""
    response = client.post(
        "/flag_prediction/",
        json={"text": "This is a test text.", "true_label": 0}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback recorded successfully!"}

@pytest.mark.api
def test_root_endpoint():
    """Test the root endpoint of the API"""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "AI Detection System API" in response.json()["message"]

@pytest.mark.api
def test_predict_endpoint():
    """Test the prediction endpoint"""
    # Test with valid input
    response = client.post(
        "/predict/",
        json={"text": "This is a test text for prediction."}
    )
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()["prediction"], int)
    assert isinstance(response.json()["probability"], float)

