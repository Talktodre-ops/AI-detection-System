import streamlit as st
import os
import sys
import pandas as pd
import torch
from typing import Optional
import io
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mixed_content_detector import MixedContentDetector

# Set page configuration
st.set_page_config(
    page_title="Mixed Content Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS with improved colors for better readability
st.markdown("""
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
            type=["txt"],
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
                    st.subheader("Detected Transitions")
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
                            f"""
                            <div style="padding: 12px; background-color: {bg_color}; border-radius: 5px; 
                                        margin-bottom: 10px; border-left: 4px solid {border_color};">
                                <p style="font-weight: bold; margin-bottom: 8px; color: #000000;">
                                    Transition from 
                                    <span style="color: {from_color};">{t['from']}</span> to 
                                    <span style="color: {to_color};">{t['to']}</span>
                                </p>
                                <p style="color: #333333; font-size: 15px;">"{t['segment']}"</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Display segments
                st.subheader("Segment Analysis")

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
                        f"""
                        <div class="segment {segment_class}">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold;">{label}</span>
                                <span>{confidence_text}</span>
                            </div>
                            <div style="margin-top: 0.5rem;">{segment}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()









