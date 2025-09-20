# AI Detection System

A comprehensive system for detecting AI-generated content using advanced NLP techniques. This project provides a suite of tools for data preprocessing, model training, and content analysis, all accessible through a user-friendly Streamlit web application.

## Features

-   **Advanced Text Analysis:** Utilizes a RoBERTa-based model to accurately distinguish between human-written and AI-generated text.
-   **Mixed Content Detection:** Capable of identifying and analyzing text that contains a mixture of AI and human-written content, breaking it down by segment.
-   **Multiple File Type Support:** The application can process `.txt`, `.pdf`, and `.docx` files, in addition to direct text input.
-   **RESTful API:** A FastAPI-based API is available for programmatic access to the detection model.
-   **Comprehensive Test Suite:** Includes a full suite of tests to ensure the reliability and accuracy of the system.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/AI-detection-System.git
    cd AI-detection-System
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv AI-Detection-311
    source AI-Detection-311/bin/activate  # On Windows: AI-Detection-311\Scripts\activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Web Application

To start the main application, run the following command:

```bash
streamlit run src/frontend/mixed_content_app.py
```

This will open the AI Detection System in your web browser, where you can upload files or paste text for analysis.

### Running the API

For programmatic access, you can run the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

The API documentation will be available at `http://127.0.0.1:8000/docs`.

### Running Tests

To ensure everything is working correctly, you can run the test suite:

```bash
cd src
python -m pytest tests/ --cov=src
```

## Project Structure

```
AI-detection-System/
├── .git/                     # Git directory
├── AI-Detection-311/         # Virtual environment
├── research/                 # Jupyter notebooks for research
├── src/
│   ├── __init__.py
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── database.py
│   ├── data/                   # Data preprocessing and management
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── clean_dataset.py
│   │   └── split_data.py
│   ├── frontend/               # Streamlit web application
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── mixed_content_app.py
│   ├── model/                  # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── roberta_model.py
│   │   └── evaluate_model.py
│   └── tests/                  # Test suite
│       ├── __init__.py
│       ├── test_api.py
│       ├── test_data_processing.py
│       └── test_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
