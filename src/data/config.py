"""
Configuration file for data processing scripts.
Centralizes all path definitions to ensure consistency.
"""
import os

# Get the absolute path to the project root directory
# This assumes the structure: project_root/src/data/config.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)

# Dataset files
RAW_DATASET = os.path.join(DATASETS_DIR, "AI_Human.csv")
CLEANED_DATASET = os.path.join(DATASETS_DIR, "cleaned_AI_Human.csv")
TRAIN_DATASET = os.path.join(DATASETS_DIR, "train.csv")
VAL_DATASET = os.path.join(DATASETS_DIR, "val.csv")
TEST_DATASET = os.path.join(DATASETS_DIR, "test.csv")

# Model directories
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model", "models", "distilroberta")

# Print configuration for debugging
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Datasets Directory: {DATASETS_DIR}")
    print(f"Raw Dataset: {RAW_DATASET}")
    print(f"Cleaned Dataset: {CLEANED_DATASET}")
    print(f"Train Dataset: {TRAIN_DATASET}")
    print(f"Validation Dataset: {VAL_DATASET}")
    print(f"Test Dataset: {TEST_DATASET}")
    print(f"Model Directory: {MODEL_DIR}")
