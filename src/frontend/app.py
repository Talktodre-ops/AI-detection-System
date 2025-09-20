import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import PyPDF2
from docx import Document
from typing import Optional
from lime import lime_text
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import numpy as np
import csv  # Added for CSV quoting
import requests

# API URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI-Generated Text Detector",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
        font-family: "Arial", sans-serif;
    }
    .stTextInput textarea {
        background-color: #2e2e2e;
        color: white;
        border-radius: 15px;
        padding: 15px;
    }
    .stButton>button {
        background-color: #0078D4;
        color: white;
        border-radius: 15px;
        padding: 12px 30px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #005EA5;
        transform: scale(0.95);
    }
    .result-container {
        background-color: #2e2e2e;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-bar {
        background-color: #333;
        border-radius: 10px;
        padding: 10px;
    }
    .confidence-value {
        font-size: 18px;
        font-weight: bold;
        margin: 0;
    }
    .lime-explanation {
        background-color: #333;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    .radial-progress {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin: 0 auto;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_tokenizer():
    # Fix the model path - use absolute path based on the script location
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "models", "distilroberta")
    try:
        # Use local_files_only=True to ensure it doesn't try to download from HF
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to("cuda")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

tokenizer, model = load_model_and_tokenizer()

def read_file(file: UploadedFile) -> Optional[str]:
    """Read text from PDF/TXT/DOCX files."""
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() for page in pdf_reader.pages])
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return " ".join([para.text for para in doc.paragraphs])
        else:
            st.error("Unsupported file type. Please upload PDF, TXT, or DOCX.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Define predict_proba for LIME
def predict_proba(text_list):
    """Return probabilities for all texts in text_list."""
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=512,  # Match the model's max context length
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    return probs

# Title and logo
st.title("🔍 AI-Generated Text Detector")
st.subheader("Identify AI vs. Human-Written Text with Confidence")

# Input options
st.sidebar.header("Input Options")
input_type = st.sidebar.radio(
    "Choose input type:",
    ("Upload File", "Enter Text"),
    index=0,
    key="input_type",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Additional Tools")
if st.sidebar.button("🔍 Mixed Content Detector"):
    switch_page("mixed_content_app")  # You'll need to import switch_page

col1, col2 = st.columns([4, 1])

with col1:
    if input_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            with st.spinner("Extracting text..."):
                text = read_file(uploaded_file)
            if text:
                st.subheader("Preview:")
                st.write(text[:500] + "..." if len(text) > 500 else text)
    else:
        text = st.text_area(
            "Paste text here",
            height=200,
            placeholder="Enter text to analyze...",
            label_visibility="collapsed",
        )

with col2:
    st.empty()  # Space for buttons

# Initialize variables
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

detect_button = st.button("Detect", use_container_width=True, type="primary")

if detect_button:
    if not text.strip():
        st.warning("Please provide text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model(**inputs)
                st.session_state.prediction = torch.argmax(outputs.logits).item()
                st.session_state.confidence = torch.max(torch.softmax(outputs.logits, dim=1)).item()

if st.session_state.prediction is not None:
    confidence_color = (
        "#4CAF50" if st.session_state.confidence >= 0.8 else
        "#FFA500" if st.session_state.confidence >= 0.5 else
        "#FF5722"
    )

    # Display results
    st.markdown(
        f"""
        <div class="result-container">
            <h3 style="color: {'#FF4B4B' if st.session_state.prediction else '#34C759'}; font-weight: bold;">
                {f"{'AI-Generated' if st.session_state.prediction else 'Human-Written'}"} 
                <small style="color: #808080; font-size: 14px;">(Confidence)</small>
            </h3>
            <div class="confidence-bar">
                <div class="confidence-value" style="color: {confidence_color};">
                    {f"{st.session_state.confidence:.0%}"}
                </div>
                <div class="radial-progress" style="
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    background: conic-gradient({confidence_color} 0% {st.session_state.confidence*360}deg, #444 {st.session_state.confidence*360}deg);
                    margin: 0 auto;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.confidence < 0.6:
        st.warning("Confidence is low. Prediction may be uncertain.")

    # Fact-check button
    if st.button("Verify Facts", use_container_width=True):
        with st.spinner("Fact-checking..."):
            try:
                response = requests.post(f"{API_URL}/fact-check", json={"text": text})
                if response.status_code == 200:
                    results = response.json()
                    st.session_state.fact_check_results = results
                else:
                    st.error(f"Fact-checking failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the fact-checking service: {e}")

    if 'fact_check_results' in st.session_state:
        results = st.session_state.fact_check_results
        if results.get("claims"):
            with st.expander("Fact-Checking Results", expanded=True):
                for c in results["claims"]:
                    st.markdown(f"**Claim:** {c['claim']}")
                    st.markdown(f"- Verdict: `{c['verdict']}`  | Confidence: `{c['confidence']:.2f}`")
                    for ev in c.get("citations", []):
                        st.markdown(f"  - [{ev.get('title')}]({ev.get('url')}) — {ev.get('stance')} (score {ev.get('score')})")

    # Explain button
    if st.button("Explain Prediction", use_container_width=True, type="secondary"):
        with st.spinner("Generating explanation..."):
            explainer = lime_text.LimeTextExplainer(
                class_names=["Human-Written", "AI-Generated"],
                split_by="sentence",
            )
            exp = explainer.explain_instance(text, predict_proba, num_features=10)
            
            st.markdown(
                f"""
                <div class="lime-explanation">
                    <h4>Key Influencing Factors:</h4>
                    <div style="padding: 15px;">
                        {exp.as_html()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Save prediction to CSV (moved inside the detect_button block)
    save_path = "src/data/datasets/user_submissions.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        "text": [text],
        "predicted_label": [st.session_state.prediction],
        "confidence": [st.session_state.confidence]
    }
    df = pd.DataFrame(data)
    
    try:
        if os.path.exists(save_path):
            df.to_csv(save_path, mode="a", header=False, index=False)
        else:
            df.to_csv(save_path, index=False)
        st.success(f"Prediction saved to {save_path} for future retraining!")
    except PermissionError:
        st.error("Permission denied. Could not save prediction data.")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: #808080;">
        <p>Developed by Talktodre-ops | 
           <a href="https://github.com/Talktodre-ops/AI-generated-content-detection" 
              style="color: #0078D4; text-decoration: none;">
              GitHub Repo
           </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)