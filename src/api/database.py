"""
Database module for AI Detection System.
Currently stores feedback data to CSV for model retraining.
"""
import os
import pandas as pd
from typing import Dict


def save_feedback(feedback_data: Dict[str, any], filepath: str = "datasets/feedback.csv") -> bool:
    """
    Save user feedback for incorrect predictions.

    Args:
        feedback_data: Dictionary containing text and true_label
        filepath: Path to feedback CSV file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame([feedback_data])
        df.to_csv(filepath, mode="a", header=not os.path.exists(filepath), index=False)
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False
