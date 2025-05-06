import pandas as pd
import numpy as np

def inspect_dataset(file_path):
    """Inspect a dataset for potential issues."""
    print(f"Inspecting dataset: {file_path}")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Check column names
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"Missing values:\n{missing}")
    
    # Check data types
    print(f"Data types:\n{df.dtypes}")
    
    # Check for non-string values in 'cleaned_text'
    if 'cleaned_text' in df.columns:
        non_string = df['cleaned_text'].apply(lambda x: not isinstance(x, str) if not pd.isna(x) else True)
        non_string_count = non_string.sum()
        print(f"Non-string values in 'cleaned_text': {non_string_count}")
        
        if non_string_count > 0:
            print("\nSample of non-string values:")
            non_string_samples = df[non_string]['cleaned_text'].head(5)
            for i, sample in enumerate(non_string_samples):
                print(f"  {i+1}. Type: {type(sample)}, Value: {sample}")
    
    # Check for very short or empty strings
    if 'cleaned_text' in df.columns:
        short_text = df['cleaned_text'].apply(lambda x: len(str(x)) < 5 if not pd.isna(x) else True)
        short_text_count = short_text.sum()
        print(f"Very short texts (< 5 chars): {short_text_count}")
        
        if short_text_count > 0:
            print("\nSample of short texts:")
            short_samples = df[short_text]['cleaned_text'].head(5)
            for i, sample in enumerate(short_samples):
                print(f"  {i+1}. Length: {len(str(sample))}, Value: {sample}")
    
    # Check label distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"\nLabel distribution:\n{label_counts}")
    
    return df

if __name__ == "__main__":
    # Inspect test dataset
    #test_path = "../data/datasets/test.csv"
    #inspect_dataset(test_path)
    
    # You can also inspect train and validation datasets
    train_path = "../data/datasets/train.csv"
    # val_path = "../data/datasets/val.csv"
    inspect_dataset(train_path)
    # inspect_dataset(val_path)