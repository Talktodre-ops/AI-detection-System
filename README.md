# AI Detection System

A comprehensive system for detecting AI-generated content using advanced NLP techniques.

## Features

- Text preprocessing and cleaning
- RoBERTa-based model for AI content detection
- API for integration with other systems
- Frontend interface for easy use
- Comprehensive test suite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-detection-System.git
cd AI-detection-System
```

2. Create and activate a virtual environment:
```bash
python -m venv AI-Detection-311
source AI-Detection-311/bin/activate  # On Windows: AI-Detection-311\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```bash
python -m src.data.clean_dataset input_data.csv output_data.csv
```

### Model Training

```bash
python -m src.models.roberta_model
```

### Running the API

```bash
uvicorn src.api.main:app --reload
```

### Running the Frontend

```bash
streamlit run src.frontend.app
```

## Testing

Run the test suite:

```bash
cd src
python -m pytest tests/ --cov=src
```

## Project Structure

```
src/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── database.py
├── data/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clean_dataset.py
│   └── split_data.py
├── models/
│   ├── __init__.py
│   ├── roberta_model.py
│   └── evaluate_model.py
├── frontend/
│   ├── __init__.py
│   └── app.py
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_data_processing.py
    └── test_model.py
```

## License

[MIT License](LICENSE)