# AI Text Detector: User Guide

## Introduction

Welcome to the AI Text Detector! This application helps you determine whether text is human-written or AI-generated using a state-of-the-art DistilRoBERTa model. This guide will help you use the application effectively and understand its outputs.

## Key Features

### High-Capacity Text Processing
Unlike most AI detection tools that limit analysis to 200-300 words, our system can process:
- Up to 50,000 characters (approximately 10-15 pages)
- Full essays, articles, and reports
- Long-form content without splitting

### Segment-Level Analysis
The system breaks down text into:
- Individual sentences or paragraphs
- Analyzes each segment independently
- Identifies transitions between AI and human writing

## Getting Started

### Installation

1. Ensure you have Python 3.7+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run src/frontend/app.py
   ```
   
   For GPU acceleration (recommended for faster processing):
   ```bash
   streamlit run src/frontend/ai_detector_gpu.py
   ```

### Interface Overview

The application provides a simple interface with:
- Text input options (direct paste or file upload)
- Detection button
- Results display
- Explanation feature

## Using the Application

### Input Methods

#### Option 1: Paste Text
1. Select "Paste Text" from the input type selector
2. Enter or paste the text you want to analyze
3. Click "Detect"

#### Option 2: Upload File
1. Select "Upload File" from the input type selector
2. Upload a PDF, TXT, or DOCX file
3. The text will be extracted automatically
4. Click "Detect"

### Understanding Results

After clicking "Detect," you'll see:

1. **Classification Result**:
   - "Human-Written" (green) or "AI-Generated" (red)

2. **Confidence Score**:
   - Displayed as a percentage
   - Higher values indicate greater confidence
   - Color-coded: Green (high confidence), Yellow (medium), Red (low)

3. **Warning Messages**:
   - If confidence is below 60%, a warning will appear

### Explanation Feature

To understand why the model made its prediction:

1. Click "Explain Prediction" after getting a result
2. The system will highlight text portions that influenced the decision
3. Red highlights indicate features suggesting AI-generated text
4. Blue highlights indicate features suggesting human-written text
5. The intensity of the color shows the importance of that feature

## Interpreting Results

### High Confidence Results (>80%)
- Generally reliable
- Still consider context and known limitations

### Medium Confidence Results (50-80%)
- Exercise caution
- Review the explanation to understand the decision
- Consider the text's context and purpose

### Low Confidence Results (<50%)
- Treat with skepticism
- The model is uncertain
- Human judgment is especially important

## Known Limitations

### AI Text Mimicking Human Imperfections
The model may misclassify AI-generated text that deliberately includes:
- Spelling errors
- Grammar mistakes
- Informal language
- Simple sentence structures

### Context Sensitivity
Certain contexts may be more challenging:
- Student writing
- Very short texts
- Highly technical content
- Creative writing

## Best Practices

1. **Use Multiple Checks**
   - For important decisions, don't rely solely on the detector
   - Consider using multiple detection tools
   - Apply human judgment

2. **Check Explanations**
   - Always review the explanation for insights
   - Look for patterns in what the model considers "human" or "AI"

3. **Consider Text Length**
   - Longer texts generally yield more reliable results
   - Very short texts may have lower confidence

4. **Be Aware of Updates**
   - AI generation technology evolves rapidly
   - Check for model updates regularly

## Feedback and Improvement

Your feedback helps improve the system:

1. If you believe a classification is incorrect, you can:
   - Note the text and classification
   - Submit through the API endpoint: `/flag_prediction/`

2. All predictions are automatically saved (anonymously) to help improve future versions of the model.

## Privacy and Data Usage

- Submitted text is processed locally
- Predictions are saved without personal identifiers
- No full texts are stored in logs
- Data is used only for model improvement

## Troubleshooting

### Common Issues

1. **File Upload Problems**
   - Ensure file is PDF, TXT, or DOCX format
   - Check file isn't corrupted
   - Try copying text directly instead

2. **Slow Processing**
   - Very long documents may take time to process
   - Consider using the GPU version for faster processing
   - Try processing smaller sections separately

3. **Permission Errors**
   - Ensure the application has write permissions to save results
   - Run as administrator if needed

## Getting Help

For additional assistance:
- Check the GitHub repository for updates
- Submit issues through GitHub
- Consult the technical documentation for advanced usage

