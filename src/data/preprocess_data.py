import pandas as pd
from src.data.preprocessing import clean_text  # Adjust path if needed

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
    raw_path = "src/data/datasets/AI_Human.csv"  # Raw dataset path
    cleaned_path = "src/data/datasets/cleaned_AI_Human.csv"
    preprocess_data(raw_path, cleaned_path)