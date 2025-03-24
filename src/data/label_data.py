import os
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

def label_data(data_path, output_path):
    """
    Load and label the dataset and save the labeled data to a CSV file.

    Args:
        data_path (str): The path to the dataset CSV file.
        output_path (str): The path to save the labeled data CSV file.
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Ensure the dataset has the required columns
    if 'text' not in data.columns or 'generated' not in data.columns:
        logging.error("Dataset does not have the required columns: 'text' and 'generated'")
        return

    # Rename the 'generated' column to 'label' for consistency
    data.rename(columns={'generated': 'label'}, inplace=True)

    # Save the labeled data to a CSV file
    data.to_csv(output_path, index=False)
    logging.info(f"Labeled data saved to: {output_path}")

if __name__ == "__main__":
    data_path = "datasets/plagiarism_sources/arxiv_papers.xml"  # Assuming this is the path to your dataset
    output_path = "datasets/labeled_data.csv"
    label_data(data_path, output_path)