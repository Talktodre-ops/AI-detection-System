import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/preprocessing.py",
    "src/data/label_data.py",
    "src/data/split_data.py",
    "src/data/data_collection.md",
    "src/models/__init__.py",
    "src/models/roberta_model.py",
    "src/models/plagiarism_detection.py",
    "src/models/explainability.py",
    "src/models/retrain_model.py",
    "src/models/model_development.md",
    "src/api/__init__.py",
    "src/api/main.py",
    "src/api/database.py",
    "src/api/api_documentation.md",
    "src/frontend/__init__.py",
    "src/frontend/app.py",
    "src/frontend/frontend_documentation.md",
    "src/tests/__init__.py",
    "src/tests/test_models.py",
    "src/tests/testing_results.md",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "deploy.bat",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")