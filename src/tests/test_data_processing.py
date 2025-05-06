import pytest
import os
import pandas as pd
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your data processing functions here
# For example:
# from data.preprocess_data import clean_text, preprocess_dataset

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        "text": [
            "This is a sample text with some punctuation!",
            "Another example with UPPERCASE letters.",
            "Numbers like 123 and special chars @#$.",
            ""
        ],
        "label": [0, 1, 0, 1]
    })

@pytest.mark.data
def test_data_loading():
    """Test that datasets can be loaded"""
    # Test paths to your datasets
    train_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "data", "datasets", "train.csv")
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "datasets", "test.csv")
    
    # Check if files exist
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    # If files don't exist, check for cleaned versions
    if not train_exists:
        train_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "data", "datasets", "train_cleaned.csv")
        train_exists = os.path.exists(train_path)
    
    if not test_exists:
        test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "data", "datasets", "test_cleaned.csv")
        test_exists = os.path.exists(test_path)
    
    # Assert that at least one dataset exists
    assert train_exists or test_exists, "No dataset files found"
    
    # If datasets exist, check that they can be loaded
    if train_exists:
        df_train = pd.read_csv(train_path)
        assert not df_train.empty, "Train dataset is empty"
        assert "label" in df_train.columns, "Train dataset missing 'label' column"
    
    if test_exists:
        df_test = pd.read_csv(test_path)
        assert not df_test.empty, "Test dataset is empty"
        assert "label" in df_test.columns, "Test dataset missing 'label' column"

# Uncomment and adapt these tests if you have the corresponding functions
"""
@pytest.mark.data
def test_text_cleaning(sample_data):
    # Test the text cleaning function
    cleaned_texts = [clean_text(text) for text in sample_data["text"]]
    
    # Check that cleaning worked as expected
    assert all(text.islower() for text in cleaned_texts if text), "Not all texts were converted to lowercase"
    assert all("@" not in text for text in cleaned_texts), "Special characters were not removed"
    assert all(text.strip() for text in cleaned_texts if text), "Texts were not properly stripped"

@pytest.mark.data
def test_dataset_preprocessing(sample_data):
    # Test the dataset preprocessing function
    processed_df = preprocess_dataset(sample_data)
    
    # Check that the processed dataframe has the expected columns
    assert "cleaned_text" in processed_df.columns, "Processed dataframe missing 'cleaned_text' column"
    assert "label" in processed_df.columns, "Processed dataframe missing 'label' column"
    
    # Check that all texts were processed
    assert not processed_df["cleaned_text"].isnull().any(), "Some texts were not processed"
"""