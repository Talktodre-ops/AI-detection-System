import streamlit as st
import os
import sys
import pandas as pd
import torch
from typing import Optional, List
import io
from streamlit.runtime.uploaded_file_manager import UploadedFile
import PyPDF2
import docx
import requests
from textblob import TextBlob

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mixed_content_detector import MixedContentDetector
#from api.fact_checker import FactChecker
try:
	import spacy
	_NLP = spacy.load("en_core_web_sm")
except Exception:
	_NLP = None

FACT_ENTITY_TYPES = {
	"PERSON","ORG","GPE","LOC","NORP","FAC","PRODUCT","EVENT",
	"WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY",
	"QUANTITY","ORDINAL","CARDINAL"
}

def _score_sentence(sent):
	# Heuristic "claiminess" score
	toks = [t for t in sent if not t.is_space]
	has_verb = any(t.pos_ in ("VERB","AUX") for t in toks)
	has_cop = any(t.lemma_ in ("be","have") for t in toks)
	ents = [e for e in sent.ents if e.label_ in FACT_ENTITY_TYPES]
	has_digit = any(ch.isdigit() for ch in sent.text)
	is_question = "?" in sent.text
	length = len(toks)

	score = 0
	if has_verb: score += 1
	if has_cop: score += 0.5
	score += min(2, len(ents)) * 0.8
	if has_digit: score += 0.5
	if 6 <= length <= 40: score += 0.5
	if is_question: score -= 2
	return score

def extract_claims(text: str, max_claims: int = 5) -> List[str]:
	text = (text or "").strip()
	if not text:
		return []

	# Prefer spaCy sentence-based claim detection
	if _NLP is not None:
		doc = _NLP(text)
		candidates = []
		for sent in doc.sents:
			s = sent.text.strip()
			if not s: 
				continue
			score = _score_sentence(sent)
			if score > 1:  # threshold for likely factual statement
				candidates.append((score, s))

		# Dedup normalized sentences and keep top-k by score
		seen = set()
		claims = []
		for _, s in sorted(candidates, key=lambda x: x[0], reverse=True):
			key = " ".join(s.lower().split())
			if key not in seen:
				seen.add(key)
				claims.append(s[:240])
			if len(claims) >= max_claims:
				break

		if claims:
			return claims

	# Fallback to TextBlob noun phrases + sentence fallback
	blob = TextBlob(text)
	cands = list({p.strip() for p in blob.noun_phrases if len(p.split()) > 1})
	if len(cands) < 2:
		cands.extend([s.strip() for s in text.split(".") if len(s.split()) > 3][:3])

	seen = set()
	claims = []
	for c in cands:
		c = c[:240]
		k = c.lower()
		if k not in seen:
			seen.add(k)
			claims.append(c)
	return claims[:max_claims]

# Set page configuration
st.set_page_config(
    page_title="AI Content Detector",
    page_icon="🔍",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# Custom CSS with improved colors for better readability
st.markdown(u"""
<style>
.main {
    padding: 2rem;
}
.stTextArea textarea {
    height: 300px;
}
.result-container {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.segment {
    padding: 0.75rem;
    border-radius: 0.25rem;
    margin-bottom: 0.5rem;
    color: #333333;
    font-size: 15px;
}
.segment-ai {
    background-color: #e6f7ff;  /* Lighter, more readable blue */
    border-left: 4px solid #1890ff;  /* Clearer blue border */
}
.segment-human {
    background-color: #f6ffed;  /* Lighter, more readable green */
    border-left: 4px solid #52c41a;  /* Clearer green border */
}
.segment-neutral {
    background-color: #f5f5f5;  /* Light gray background */
    border-left: 4px solid #d9d9d9;  /* Lighter gray border */
}
.confidence-bar {
    display: flex;
    align-items: center;
    margin-top: 1rem;
}
.confidence-value {
    font-weight: bold;
    margin-right: 1rem;
}
</style>
""", unsafe_allow_html=True)

def read_file(file: UploadedFile) -> Optional[str]:
    """Read text from uploaded files."""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                # Clean up PDF text extraction issues
                if page_text:
                    # Replace multiple newlines with single space
                    page_text = ' '.join(page_text.split())
                    # Fix common PDF spacing issues
                    page_text = page_text.replace('  ', ' ')  # Double spaces
                    page_text = page_text.replace(' .', '.')  # Space before period
                    page_text = page_text.replace(' ,', ',')  # Space before comma
                    page_text = page_text.replace(' :', ':')  # Space before colon
                    page_text = page_text.replace(' ;', ';')  # Space before semicolon
                    text += page_text + " "
            return text.strip()
        elif file_extension == ".docx":
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            # Read as bytes
            bytes_data = file.read()
            
            # Try to decode as UTF-8
            try:
                return bytes_data.decode("utf-8")
            except UnicodeDecodeError:
                # If UTF-8 fails, try other common encodings
                try:
                    return bytes_data.decode("latin-1")
                except:
                    st.error("Could not decode file. Please ensure it's a text file.")
                    return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


@st.cache_resource
def load_detector():
    """Load the mixed content detector model."""
    # Use the same model path as in the main app
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "models", "distilroberta")
    return MixedContentDetector(model_dir=model_dir)

def main():
    st.title("🔍 Mixed Content Detector")
    st.subheader("Identify text with mixed AI and human-written content")
    
    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.info(
        "This tool analyzes text to detect if it contains a mix of AI-generated and human-written content. "
        "It breaks down the text into segments and analyzes each one independently. "
        "**Capable of processing up to 50,000 characters** (approximately 10-15 pages), "
        "significantly more than most similar tools that limit to 200 words."
    )
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Using device: {device.upper()}")
    if device == "cuda":
        st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load the detector
    with st.spinner("Loading model..."):
        detector = load_detector()
    
    # Input options
    st.sidebar.header("Input Options")
    input_type = st.sidebar.radio(
        "Choose input type:",
        ("Enter Text", "Upload File"),
        index=0,
        key="input_type",
    )
    
    text = None
    
    if input_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a text file",
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            with st.spinner("Extracting text..."):
                text = read_file(uploaded_file)
            if text:
                st.subheader("Preview:")
                st.write(text[:500] + "..." if len(text) > 500 else text)
                st.caption(f"Character count: {len(text)} characters")
    else:
        text = st.text_area(
            "Paste text here",
            height=200,
            placeholder="Enter text to analyze...",
        )
        if text:
            st.caption(f"Character count: {len(text)} characters")
    
    # Analysis options
    st.sidebar.header("Analysis Options")
    segment_type = st.sidebar.selectbox(
        "Segment by:",
        ["sentence", "paragraph"],
        index=0
    )
    
    # Advanced mode toggle
    advanced_mode = st.sidebar.checkbox("Advanced Mode", value=False)
    
    # Only show confidence threshold slider in advanced mode
    if advanced_mode:
        confidence_threshold = st.sidebar.slider(
            "Confidence threshold:",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Higher values mean more confident predictions but may miss subtle transitions."
        )
    else:
        # Use auto-adjustment for normal mode
        confidence_threshold = None  # Will be set during analysis
    
    # Analyze button
    analyze_button = st.button("Analyze", use_container_width=True, type="primary")
    
    if analyze_button:
        if not text or not text.strip():
            st.warning("Please provide text to analyze.")
        else:
            with st.spinner("Analyzing text for mixed content..."):
                # Set segment type
                detector.segment_type = segment_type
                
                # Auto-adjust confidence threshold if not in advanced mode
                if confidence_threshold is None:
                    text_length = len(text)
                    if text_length < 500:
                        confidence_threshold = 0.8
                    elif text_length < 2000:
                        confidence_threshold = 0.7
                    else:
                        confidence_threshold = 0.6
                
                # Analyze the text
                analysis_results = detector.analyze_text(
                    text,
                    threshold=confidence_threshold
                )
                
                # Store results in session state
                st.session_state.analysis_results = analysis_results
                st.session_state.confidence_threshold = confidence_threshold
                st.session_state.segment_type = segment_type

    # Display analysis results if they exist in session state
    if 'analysis_results' in st.session_state:
        analysis_results = st.session_state.analysis_results
        confidence_threshold = st.session_state.confidence_threshold
        segment_type = st.session_state.segment_type
        
        # Display results
        st.subheader("Analysis Results")

        # Overall statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "AI Content",
                f"{analysis_results['ai_percentage']:.1f}%" if analysis_results['ai_percentage'] is not None else "N/A"
            )

        with col2:
            st.metric(
                "Human Content",
                f"{analysis_results['human_percentage']:.1f}%" if analysis_results['human_percentage'] is not None else "N/A"
            )

        with col3:
            is_mixed = analysis_results['is_mixed']
            st.metric(
                "Mixed Content",
                "Yes" if is_mixed else "No",
                delta="Detected" if is_mixed else None
            )

        # Only display transitions if in advanced mode
        if advanced_mode and analysis_results['transitions']:
            with st.expander("🔀 Detected Transitions", expanded=False):
                for t in analysis_results['transitions']:
                    # Determine colors based on transition type
                    if t['from'] == 'Human' and t['to'] == 'AI':
                        # Transition from Human to AI
                        from_color = "#52c41a"  # Human color
                        to_color = "#1890ff"    # AI color
                        bg_color = "#f6ffed"    # Light human background
                        border_color = "#1890ff"  # AI border
                    else:
                        # Transition from AI to Human
                        from_color = "#1890ff"  # AI color
                        to_color = "#52c41a"    # Human color
                        bg_color = "#e6f7ff"    # Light AI background
                        border_color = "#52c41a"  # Human border
                    st.markdown(
                        f'''
                        <div style="padding: 12px; background-color: {bg_color}; border-radius: 5px; 
                                    margin-bottom: 10px; border-left: 4px solid {border_color};">
                            <p style="font-weight: bold; margin-bottom: 8px; color: #000000;">
                                Transition from 
                                <span style="color: {from_color};">{t['from']}</span> to 
                                <span style="color: {to_color};">{t['to']}</span>
                            </p>
                            <p style="color: #333333; font-size: 15px;">"{t['segment']}"</p>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

        # Display segments in collapsible section
        with st.expander("📄 Segment Analysis", expanded=False):
            for i, (segment, prediction, confidence) in enumerate(analysis_results['segment_results']):
                if prediction is None:
                    segment_class = "segment-neutral"
                    label = "Too short"
                    confidence_text = "-"
                else:
                    segment_class = "segment-ai" if prediction == 1 else "segment-human"
                    label = "AI" if prediction == 1 else "Human"
                    confidence_text = f"{confidence:.0%}"
                
                st.markdown(
                    f'''
                    <div class="segment {segment_class}">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: bold;">{label}</span>
                            <span>{confidence_text}</span>
                        </div>
                        <div style="margin-top: 0.5rem;">{segment}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

    # Fact-Check button
    if st.button("Fact-Check", use_container_width=True):
        if not text or not text.strip():
            st.warning("Please provide text to fact-check.")
        else:
            with st.spinner("Fact-checking text..."):
                try:
                    resp = requests.post(f"{API_URL}/fact-check", json={"text": text})
                    if resp.status_code == 200:
                        results = resp.json()
                        # Store fact-check results in session state
                        st.session_state.fact_check_results = results
                    else:
                        st.error(f"Fact-checking failed with status code {resp.status_code}")
                        results = None
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to fact-checking Server: {str(e)}")
                    results = None

    # Display fact-check results if they exist in session state
    if 'fact_check_results' in st.session_state:
        results = st.session_state.fact_check_results
        if results and results.get("claims"):
            with st.expander("🔍 Fact-Checking Results", expanded=False):
                for c in results["claims"]:
                    st.markdown(f"**Claim:** {c['claim']}")
                    st.markdown(f"- Verdict: `{c['verdict']}`  | Confidence: `{c['confidence']:.2f}`")
                    for ev in c.get("citations", []):
                        st.markdown(f"  - [{ev.get('title')}]({ev.get('url')}) — {ev.get('stance')} (score {ev.get('score')})")
        else:
            with st.expander("🔍 Fact-Checking Results", expanded=False):
                st.info("No claims were found to fact-check.")

    # Clear results button (optional)
    if st.sidebar.button("Clear Results"):
        if 'analysis_results' in st.session_state:
            del st.session_state.analysis_results
        if 'fact_check_results' in st.session_state:
            del st.session_state.fact_check_results
        st.rerun()

if __name__ == "__main__":
    main()