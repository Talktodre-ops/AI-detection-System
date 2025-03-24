# AI-Generated Text Detector

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction
This project detects whether text is AI-generated or human-written using a DistilRoBERTa-based model. It supports PDF, TXT, and DOCX files, and includes explainability features using LIME.

---

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/AI-generated-content-detection.git
   cd AI-generated-content-detection

   python -m venv ai-detection-env
    ai-detection-env\Scripts\activate  # On Windows

    pip install -r requirements.txt
Download the model (only required once)
python
>>> from transformers import AutoModelForSequenceClassification
>>> AutoModelForSequenceClassification.from_pretrained(
...     "distilroberta-base",
...     cache_dir="src/model/models/distilroberta"
... )
exit()

Run the App
streamlit run src/frontend/app.py

Features :
Upload PDF/TXT/DOCX files or paste text.
View predictions with confidence scores (AI/Human).
Get explanations of influential text segments using LIME.
Automatically save predictions to src/data/datasets/user_submissions.csv.
UI Preview :
App Screenshot
<!-- Replace with your screenshot link -->
AI-generated-content-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── data/
│   │   └── datasets/
│   │       ├── cleaned_AI_Human.csv  # Original dataset (ensure this exists)
│   │       └── user_submissions.csv  # Auto-saved predictions
│   ├── frontend/
│   │   ├── app.py
│   │   └── assets/
│   │       └── logo.png
│   └── model/
│       └── models/
│           └── distilroberta/  # Model weights (auto-downloaded)
├── retrain_model.py  # For retraining the model
└── ... (other files like Draw.io diagrams)

Contributing
Retraining the Model :
Add new data to src/data/datasets/user_submissions.csv.

python retrain_model.py

# Use CPU if CUDA unavailable
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu

Permission Issues :
Ensure src/data/datasets/ has write permissions.
Run the app as administrator if needed.
3. How to Add a Screenshot
Take a screenshot of your app.
Upload it to a hosting service (e.g., Imgur).
Replace https://i.imgur.com/EXAMPLE.png in the README.md with your image link.
