import os
import pandas as pd
import logging
from .preprocessing import preprocess_text
import sys
import os
import numpy as np

# Add the parent directory to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import safe_read_csv, safe_write_csv

def clean_dataset(input_path, output_path=None):
    """
    Clean a dataset by preprocessing the text column.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the cleaned CSV file. 
                                    If None, will use input_path with '_cleaned' suffix.
    
    Returns:
        pandas.DataFrame: The cleaned dataset
    """
    logging.info(f"Cleaning dataset: {input_path}")
    
    # Set default output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"
    
    # Load the dataset
    df = safe_read_csv(input_path)
    if df is None:
        logging.error(f"Failed to load dataset: {input_path}")
        return None
    
    logging.info(f"Original dataset shape: {df.shape}")
    
    # Clean the text column
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Replace NaN values with empty strings
    df['cleaned_text'] = df['cleaned_text'].fillna("")
    
    # Save the cleaned dataset
    # Convert empty strings to actual empty strings (not NaN) when saving
    success = safe_write_csv(df, output_path, index=False)
    if success:
        logging.info(f"Cleaned dataset saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Clean test dataset
    test_path = "../data/datasets/test.csv"
    cleaned_test_path = clean_dataset(test_path)
    
    # You can also clean train and validation datasets
    train_path = "../data/datasets/train.csv"
    val_path = "../data/datasets/val.csv"
    
    if os.path.exists(train_path):
        cleaned_train_path = clean_dataset(train_path)
        print(f"Train dataset cleaned: {cleaned_train_path}")
    
    if os.path.exists(val_path):
        cleaned_val_path = clean_dataset(val_path)
        print(f"Validation dataset cleaned: {cleaned_val_path}")





