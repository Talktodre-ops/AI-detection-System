import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

def split_dataset(data, train_path, val_path, test_path, test_size=0.2, val_size=0.25):
    """
    Split the cleaned dataset into train/val/test sets.
    
    Args:
        data (pd.DataFrame): The cleaned dataset with 'cleaned_text' and 'label' columns.
    """
    train_val, test = train_test_split(data, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    logging.info(f"Split saved to: {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
    data_path = "src/data/datasets/cleaned_AI_Human.csv"  # Original cleaned dataset
    data = pd.read_csv(data_path)
    split_dir = "src/data/datasets/"
    split_dataset(
        data=data,
        train_path=os.path.join(split_dir, "train.csv"),
        val_path=os.path.join(split_dir, "val.csv"),
        test_path=os.path.join(split_dir, "test.csv")
    )