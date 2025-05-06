import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

class MixedContentDetector:
    """
    Detector for mixed AI and human-written content that analyzes text at the segment level
    and identifies transitions between different writing styles.
    """
    
    def __init__(self, model_dir=None, device=None, segment_type="sentence"):
        """
        Initialize the mixed content detector.
        
        Args:
            model_dir (str): Directory containing the model files
            device (str): Device to run the model on ('cuda' or 'cpu')
            segment_type (str): Type of segmentation ('sentence' or 'paragraph')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir if model_dir else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "models", "distilroberta")
        self.segment_type = segment_type
        self.use_nltk = False
        
        # Check if NLTK punkt is available
        try:
            nltk.data.find('tokenizers/punkt')
            self.use_nltk = True
        except (LookupError, ConnectionError):
            logging.warning("NLTK punkt tokenizer not available. Using regex-based fallback.")
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
            logging.info(f"Loaded model from {self.model_dir} on {self.device}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def simple_sentence_tokenize(self, text):
        """
        Simple regex-based sentence tokenizer as fallback when NLTK is not available.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        # Split by common sentence-ending punctuation followed by whitespace
        segments = re.split(r'(?<=[.!?])\s+', text)
        return [segment.strip() for segment in segments if segment.strip()]
    
    def segment_text(self, text):
        """
        Segment text into sentences or paragraphs.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of text segments
        """
        if not text or not isinstance(text, str):
            return []
        
        if self.segment_type == "sentence":
            # Use NLTK if available, otherwise use simple tokenizer
            if self.use_nltk:
                try:
                    segments = sent_tokenize(text)
                except Exception as e:
                    logging.warning(f"NLTK sentence tokenization failed: {str(e)}. Using fallback method.")
                    segments = self.simple_sentence_tokenize(text)
            else:
                segments = self.simple_sentence_tokenize(text)
        else:
            # Split by paragraphs
            segments = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # If no paragraph breaks, fall back to sentences
            if len(segments) <= 1:
                if self.use_nltk:
                    try:
                        segments = sent_tokenize(text)
                    except Exception as e:
                        logging.warning(f"NLTK sentence tokenization failed: {str(e)}. Using fallback method.")
                        segments = self.simple_sentence_tokenize(text)
                else:
                    segments = self.simple_sentence_tokenize(text)
        
        return segments
    
    def analyze_segments(self, segments, min_segment_length=15):
        """
        Analyze individual segments for AI vs. human classification.
        
        Args:
            segments (list): List of text segments
            min_segment_length (int): Minimum character length for a segment to be analyzed
            
        Returns:
            list: List of (segment, prediction, confidence) tuples
        """
        results = []
        
        for segment in segments:
            # Skip very short segments
            if len(segment) < min_segment_length:
                results.append((segment, None, None))
                continue
                
            # Tokenize the segment
            inputs = self.tokenizer(
                segment,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
            
            results.append((segment, prediction, confidence))
        
        return results
    
    def detect_transitions(self, segment_results, threshold=0.7):
        """
        Detect transitions between AI and human content.
        
        Args:
            segment_results (list): List of (segment, prediction, confidence) tuples
            threshold (float): Confidence threshold for reliable predictions
            
        Returns:
            list: List of detected transitions with indices
        """
        transitions = []
        prev_prediction = None
        
        for i, (segment, prediction, confidence) in enumerate(segment_results):
            # Skip segments with no prediction
            if prediction is None:
                continue
                
            # Skip low-confidence predictions
            if confidence < threshold:
                continue
                
            # Check for transition
            if prev_prediction is not None and prediction != prev_prediction:
                transitions.append({
                    'index': i,
                    'from': 'Human' if prev_prediction == 0 else 'AI',
                    'to': 'Human' if prediction == 0 else 'AI',
                    'segment': segment,
                    'confidence': confidence
                })
                
            prev_prediction = prediction
        
        return transitions
    
    def analyze_text(self, text, min_segment_length=15, threshold=0.7):
        """
        Analyze text for mixed AI and human content.
        
        Args:
            text (str): Input text
            min_segment_length (int): Minimum character length for a segment to be analyzed
            threshold (float): Confidence threshold for reliable predictions
            
        Returns:
            dict: Analysis results including segments, predictions, and transitions
        """
        # Segment the text
        segments = self.segment_text(text)
        
        # Analyze each segment
        segment_results = self.analyze_segments(segments, min_segment_length)
        
        # Detect transitions
        transitions = self.detect_transitions(segment_results, threshold)
        
        # Calculate overall statistics
        valid_results = [(s, p, c) for s, p, c in segment_results if p is not None]
        if not valid_results:
            return {
                'segments': segments,
                'segment_results': segment_results,
                'transitions': transitions,
                'overall_prediction': None,
                'overall_confidence': None,
                'ai_percentage': None,
                'human_percentage': None,
                'is_mixed': False
            }
            
        ai_segments = sum(1 for _, p, _ in valid_results if p == 1)
        human_segments = sum(1 for _, p, _ in valid_results if p == 0)
        total_segments = len(valid_results)
        
        ai_percentage = (ai_segments / total_segments) * 100 if total_segments > 0 else 0
        human_percentage = (human_segments / total_segments) * 100 if total_segments > 0 else 0
        
        # Determine if content is mixed
        is_mixed = len(transitions) > 0
        
        # Overall prediction based on majority
        if ai_segments > human_segments:
            overall_prediction = 1  # AI
            overall_confidence = ai_percentage / 100
        else:
            overall_prediction = 0  # Human
            overall_confidence = human_percentage / 100
        
        return {
            'segments': segments,
            'segment_results': segment_results,
            'transitions': transitions,
            'overall_prediction': overall_prediction,
            'overall_confidence': overall_confidence,
            'ai_percentage': ai_percentage,
            'human_percentage': human_percentage,
            'is_mixed': is_mixed
        }


