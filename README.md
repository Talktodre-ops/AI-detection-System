# AI Detection System with Fact-Checking

A comprehensive AI-powered system for detecting AI-generated content and verifying factual claims using advanced NLP techniques. This project provides a complete suite of tools for content analysis, fact-checking, and mixed content detection, all accessible through user-friendly web applications and RESTful APIs.

![aidetect](https://github.com/user-attachments/assets/277aea2c-6321-4700-9a49-bd421bf8db28)

## 🚀 Key Features

### 🔍 **AI Content Detection**
- **Advanced Text Analysis:** Utilizes a DistilRoBERTa-based model with 99%+ accuracy to distinguish between human-written and AI-generated text
- **High-Capacity Processing:** Can analyze up to 50,000 characters (10-15 pages) - significantly more than most detection tools
- **GPU Acceleration:** Optimized for CUDA-enabled systems for faster processing
- **Real-time Confidence Scoring:** Provides detailed confidence metrics with visual indicators

### 🔄 **Mixed Content Detection**
- **Segment-Level Analysis:** Breaks down text into sentences or paragraphs and analyzes each independently
- **Transition Detection:** Identifies where content switches between AI and human writing
- **Advanced Mode:** Configurable confidence thresholds and detailed transition analysis
- **Visual Segmentation:** Color-coded display of AI vs human content segments

### ✅ **Fact-Checking & Verification**
- **Intelligent Claim Extraction:** Uses spaCy NLP to identify factual claims and statements
- **Multi-Source Evidence Retrieval:**
  - Google Custom Search API (primary)
  - Wikipedia API (fallback)
  - DuckDuckGo Search (fallback)
- **Natural Language Inference:** Uses RoBERTa-large-MNLI for stance detection (support/refute/neutral)
- **Source Credibility Scoring:** Prioritizes .gov, .edu, and trusted news sources
- **Citation Management:** Provides detailed citations with confidence scores

### 📁 **File Processing**
- **Multiple Format Support:** PDF, DOCX, TXT files with intelligent text extraction
- **PDF Optimization:** Advanced text cleaning for better PDF processing
- **Batch Processing:** Handle multiple files efficiently

### 🎯 **User Experience**
- **Dual Interface:** Basic AI detector and advanced mixed content detector
- **Collapsible Results:** Clean UI with expandable sections for detailed analysis
- **Session State Management:** Persistent results across different analysis types
- **Real-time Feedback:** Live progress indicators and error handling

## 🏗️ Architecture

### **Frontend Layer (Streamlit)**
- `app.py` - Basic AI detection interface
- `ai_detector_gpu.py` - GPU-optimized detection
- `mixed_content_app.py` - Advanced mixed content analysis with fact-checking

### **API Layer (FastAPI)**
- `main.py` - RESTful API endpoints
- `fact_checker.py` - Fact-checking and verification engine
- `database.py` - Data persistence layer

### **Core Engine**
- `mixed_content_detector.py` - Mixed content analysis engine
- `roberta_model.py` - Model training and inference
- `evaluate_model.py` - Model evaluation and metrics

### **Data Processing**
- `preprocessing.py` - Text preprocessing pipeline
- `clean_dataset.py` - Dataset cleaning utilities
- `split_data.py` - Train/validation/test splitting

## 📊 Model Performance

- **Accuracy:** 99.02%
- **Precision:** 97.63%
- **Recall:** 99.79%
- **F1-Score:** 98.70%
- **Test Dataset:** 97,445 samples
- **Misclassification Rate:** 0.35% (344 out of 97,445)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Talktodre-ops/AI-detection-System.git
cd AI-detection-System
```

2. **Create and activate virtual environment:**
```bash
python -m venv AI-Detection-311
# Windows
AI-Detection-311\Scripts\activate
# Linux/Mac
source AI-Detection-311/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model (for fact-checking):**
```bash
python -m spacy download en_core_web_sm
```

5. **Download NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

## 🚀 Usage

### Web Applications

#### Basic AI Detection
```bash
streamlit run src/frontend/app.py
```

#### GPU-Optimized Detection
```bash
streamlit run src/frontend/ai_detector_gpu.py
```

#### Advanced Mixed Content Detection with Fact-Checking
```bash
streamlit run src/frontend/mixed_content_app.py
```

### API Server

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

API Documentation: `http://127.0.0.1:8000/docs`

### Environment Variables (Optional)

For enhanced fact-checking capabilities:
```bash
# Google Custom Search (optional)
export GOOGLE_API_KEY="your_api_key"
export GOOGLE_CSE_ID="your_search_engine_id"

# Alternative NLI model (optional)
export NLI_MODEL_NAME="distilroberta-base-mnli"
```

## 📡 API Endpoints

### Core Detection
- `POST /predict/` - Basic AI detection
- `POST /fact-check/` - Fact-checking and verification

### Data Management
- `POST /flag_prediction/` - Submit feedback for model improvement
- `GET /` - Health check

### Example API Usage
```python
import requests

# AI Detection
response = requests.post("http://127.0.0.1:8000/predict/",
                        json={"text": "Your text here"})

# Fact-Checking
response = requests.post("http://127.0.0.1:8000/fact-check/",
                        json={"text": "The Eiffel Tower is in Paris."})
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd src
python -m pytest tests/ --cov=src
```

## 📁 Project Structure

```
AI-detection-System/
├── .github/workflows/          # CI/CD workflows
├── research/                   # Research notebooks
├── src/
│   ├── api/                   # FastAPI backend
│   │   ├── main.py           # API endpoints
│   │   ├── fact_checker.py   # Fact-checking engine
│   │   └── database.py       # Data persistence
│   ├── data/                  # Data processing
│   │   ├── preprocessing.py  # Text preprocessing
│   │   ├── clean_dataset.py  # Dataset cleaning
│   │   └── split_data.py     # Data splitting
│   ├── frontend/             # Streamlit applications
│   │   ├── app.py           # Basic detector
│   │   ├── ai_detector_gpu.py # GPU-optimized
│   │   └── mixed_content_app.py # Advanced detector
│   ├── model/                # ML models and training
│   │   ├── mixed_content_detector.py
│   │   ├── roberta_model.py
│   │   ├── evaluate_model.py
│   │   └── analysis/         # Model evaluation results
│   ├── tests/                # Test suite
│   └── utils/                # Utility functions
├── .gitignore
├── .pylintrc
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Model Settings
- **Base Model:** DistilRoBERTa
- **Max Sequence Length:** 512 tokens
- **Confidence Threshold:** 0.7 (configurable)
- **Segment Types:** Sentence or paragraph

### Fact-Checking Settings
- **Max Claims:** 5 per analysis
- **Evidence Sources:** Google CSE, Wikipedia, DuckDuckGo
- **NLI Model:** RoBERTa-large-MNLI
- **Confidence Threshold:** 0.3

## 🎯 Use Cases

### Educational
- **Academic Integrity:** Detect AI-generated essays and assignments
- **Research Validation:** Verify factual claims in academic papers
- **Content Analysis:** Analyze mixed human-AI content in educational materials

### Professional
- **Content Moderation:** Identify AI-generated content in user submissions
- **Fact-Checking:** Verify claims in news articles and reports
- **Quality Assurance:** Ensure content authenticity in professional documents

### Research
- **Model Evaluation:** Test and compare different AI detection methods
- **Dataset Analysis:** Analyze large text datasets for AI content
- **Performance Benchmarking:** Evaluate detection accuracy across different content types

## 🔍 Advanced Features

### Mixed Content Analysis
- **Transition Detection:** Identifies where content switches between AI and human writing
- **Confidence Scoring:** Provides detailed confidence metrics for each segment
- **Visual Representation:** Color-coded display of analysis results

### Fact-Checking Pipeline
1. **Claim Extraction:** Uses NLP to identify factual statements
2. **Evidence Retrieval:** Searches multiple credible sources
3. **Stance Detection:** Determines if evidence supports, refutes, or is neutral
4. **Confidence Scoring:** Provides reliability metrics for each claim

### Performance Optimization
- **GPU Acceleration:** CUDA support for faster processing
- **Caching:** Intelligent caching of model predictions
- **Batch Processing:** Efficient handling of multiple documents

## 🚨 Limitations

- **Model Training Data:** Performance depends on training data quality and recency
- **Evolving AI:** New AI models may require retraining
- **Context Sensitivity:** Performance may vary with different writing styles
- **Fact-Checking Accuracy:** Depends on available evidence and source quality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the DistilRoBERTa model
- **Streamlit** for the web interface framework
- **FastAPI** for the API framework
- **spaCy** for NLP processing
- **Google** for Custom Search API

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues:** [Create an issue](https://github.com/Talktodre-ops/AI-detection-System/issues)
- **Documentation:** Check the `/docs` folder for detailed guides
- **Email:** Contact the maintainers for direct support

---

**Developed by:** [Talktodre-ops](https://github.com/Talktodre-ops)
**Version:** 2.0.0
**Last Updated:** January 2025
