import pandas as pd
import os

# Use direct import since preprocessing.py is in the same directory
from preprocessing import clean_text

def preprocess_data(raw_path, cleaned_path):
    """
    Clean and preprocess the raw dataset.
    
    Args:
        raw_path (str): Path to the raw CSV file.
        cleaned_path (str): Path to save the cleaned CSV file.
    """
    data = pd.read_csv(raw_path)
    data['cleaned_text'] = data['text'].apply(clean_text)
    data = data.drop(columns=['text'])  # Drop original text column
    
    # Rename columns if needed (e.g., 'generated' → 'label')
    if 'generated' in data.columns:
        data.rename(columns={'generated': 'label'}, inplace=True)
    
    data.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")

if __name__ == "__main__":
    # Use relative paths based on the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, "datasets", "AI_Human.csv")
    cleaned_path = os.path.join(script_dir, "datasets", "cleaned_AI_Human.csv")
    
    # Ensure the datasets directory exists
    os.makedirs(os.path.join(script_dir, "datasets"), exist_ok=True)
    
    preprocess_data(raw_path, cleaned_path)
