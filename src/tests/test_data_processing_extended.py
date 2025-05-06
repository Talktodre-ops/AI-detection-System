import os
import pandas as pd
import pytest
import sys
import numpy as np

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import preprocess_text
from data.clean_dataset import clean_dataset

@pytest.mark.data
def test_preprocess_text():
    """Test the text preprocessing function"""
    # Test with normal text
    text = "Hello, World! This is a test."
    processed = preprocess_text(text)
    assert processed == "hello world this is a test"
    
    # Test with extra spaces
    text = "  Extra  spaces  "
    processed = preprocess_text(text)
    assert processed == "extra spaces"
    
    # Test with special characters
    text = "Special@#$%^&*()characters"
    processed = preprocess_text(text)
    assert processed == "special characters"
    
    # Test with empty string
    text = ""
    processed = preprocess_text(text)
    assert processed == ""
    
    # Test with None
    text = None
    processed = preprocess_text(text)
    assert processed == ""

@pytest.mark.data
def test_clean_dataset():
    """Test the dataset cleaning function"""
    # Create a small test dataset
    test_data = pd.DataFrame({
        "text": [
            "Hello, world!",
            "  Extra  spaces  ",
            "Special@#$%^&*()characters",
            ""  # Empty string
        ],
        "label": [0, 1, 0, 1]
    })

    # Save to a temporary CSV
    temp_test_path = "temp_test_clean.csv"
    test_data.to_csv(temp_test_path, index=False)

    try:
        # Clean the dataset
        output_path = "temp_test_clean_output.csv"
        cleaned_df = clean_dataset(temp_test_path, output_path)
        
        # Check if the output file exists
        assert os.path.exists(output_path)
        
        # Check that all rows are present in the returned DataFrame
        assert len(cleaned_df) == len(test_data)
        
        # Check that the text was cleaned correctly in the returned DataFrame
        assert cleaned_df.loc[0, 'cleaned_text'] == "hello world"
        assert cleaned_df.loc[1, 'cleaned_text'] == "extra spaces"
        assert cleaned_df.loc[2, 'cleaned_text'] == "special characters"
        assert cleaned_df.loc[3, 'cleaned_text'] == ""
        
        # Load the cleaned data from file to verify it was saved correctly
        # Use na_filter=False to read empty strings as empty strings, not NaN
        cleaned_data = pd.read_csv(output_path, na_filter=False)
        
        # Check that all rows are present
        assert len(cleaned_data) == len(test_data)
        
        # Check that the text was cleaned correctly
        assert cleaned_data.loc[0, 'cleaned_text'] == "hello world"
        assert cleaned_data.loc[1, 'cleaned_text'] == "extra spaces"
        assert cleaned_data.loc[2, 'cleaned_text'] == "special characters"
        assert cleaned_data.loc[3, 'cleaned_text'] == ""
        
    finally:
        # Clean up temporary files
        for file in [temp_test_path, output_path]:
            if os.path.exists(file):
                os.remove(file)

@pytest.mark.data
def test_clean_dataset_with_missing_values():
    """Test the dataset cleaning function with missing values"""
    # Create a small test dataset with NaN values
    test_data = pd.DataFrame({
        "text": [
            "Hello, world!",
            None,
            pd.NA,
            float('nan')
        ],
        "label": [0, 1, 0, 1]
    })

    # Save to a temporary CSV
    temp_test_path = "temp_test_missing.csv"
    test_data.to_csv(temp_test_path, index=False)

    try:
        # Clean the dataset
        output_path = "temp_test_missing_output.csv"
        cleaned_df = clean_dataset(temp_test_path, output_path)
        
        # Check if the output file exists and has the right content
        assert os.path.exists(output_path)
        
        # Check the returned DataFrame
        assert len(cleaned_df) == len(test_data)
        assert cleaned_df.loc[0, 'cleaned_text'] == "hello world"
        assert cleaned_df.loc[1, 'cleaned_text'] == ""
        assert cleaned_df.loc[2, 'cleaned_text'] == ""
        assert cleaned_df.loc[3, 'cleaned_text'] == ""
        
        # Load the cleaned data from file
        # Use na_filter=False to read empty strings as empty strings, not NaN
        cleaned_data = pd.read_csv(output_path, na_filter=False)
        
        # Check that all rows are present
        assert len(cleaned_data) == len(test_data)
        
        # Check that empty values were handled correctly
        assert cleaned_data.loc[0, 'cleaned_text'] == "hello world"
        assert cleaned_data.loc[1, 'cleaned_text'] == ""
        assert cleaned_data.loc[2, 'cleaned_text'] == ""
        assert cleaned_data.loc[3, 'cleaned_text'] == ""
        
    finally:
        # Clean up temporary files
        for file in [temp_test_path, output_path]:
            if os.path.exists(file):
                os.remove(file)

