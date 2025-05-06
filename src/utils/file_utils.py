"""
Utility functions for file operations
"""
import os
import pandas as pd
import logging
import numpy as np

def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary
    
    Args:
        directory (str): Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def safe_read_csv(file_path, **kwargs):
    """
    Safely read a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
        
    Returns:
        pandas.DataFrame or None: The loaded DataFrame, or None if loading failed
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path, **kwargs)
        logging.info(f"Successfully loaded {file_path} with {len(df)} rows")
        return df
        
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return None

def safe_write_csv(df, file_path, **kwargs):
    """
    Safely write a pandas DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): The DataFrame to save
        file_path (str): Path to save the CSV file
        **kwargs: Additional arguments to pass to df.to_csv()
        
    Returns:
        bool: True if saving was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Replace NaN values with empty strings before saving
        df_to_save = df.copy()
        for col in df_to_save.select_dtypes(include=['object']).columns:
            df_to_save[col] = df_to_save[col].fillna("")
            
        # Save the DataFrame
        df_to_save.to_csv(file_path, **kwargs)
        logging.info(f"Successfully saved {len(df)} rows to {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving to {file_path}: {str(e)}")
        return False
