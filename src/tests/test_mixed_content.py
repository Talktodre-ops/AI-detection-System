import os
import sys
import unittest
import re

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.mixed_content_detector import MixedContentDetector

class TestMixedContentDetector(unittest.TestCase):
    
    def setUp(self):
        # Initialize the detector
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "models", "distilroberta")
        self.detector = MixedContentDetector(model_dir=model_dir)
    
    def test_simple_sentence_tokenize(self):
        """Test the fallback sentence tokenizer."""
        text = "This is sentence one. This is sentence two. And this is the third!"
        segments = self.detector.simple_sentence_tokenize(text)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0], "This is sentence one.")
        self.assertEqual(segments[1], "This is sentence two.")
        self.assertEqual(segments[2], "And this is the third!")
    
    def test_segment_text_sentences(self):
        """Test text segmentation by sentences."""
        text = "This is sentence one. This is sentence two. And this is the third."
        segments = self.detector.segment_text(text)
        self.assertEqual(len(segments), 3)
    
    def test_segment_text_paragraphs(self):
        """Test text segmentation by paragraphs."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        self.detector.segment_type = "paragraph"
        segments = self.detector.segment_text(text)
        self.assertEqual(len(segments), 3)
    
    def test_analyze_segments(self):
        """Test segment analysis."""
        segments = ["This is a test sentence.", "Another test sentence."]
        results = self.detector.analyze_segments(segments)
        self.assertEqual(len(results), 2)
        for segment, prediction, confidence in results:
            self.assertIn(segment, segments)
            self.assertIn(prediction, [0, 1, None])
            if prediction is not None:
                self.assertTrue(0 <= confidence <= 1)
    
    def test_detect_transitions(self):
        """Test transition detection."""
        # Create mock segment results
        segment_results = [
            ("Segment 1", 0, 0.9),  # Human with high confidence
            ("Segment 2", 0, 0.8),  # Human with high confidence
            ("Segment 3", 1, 0.9),  # AI with high confidence
            ("Segment 4", 1, 0.7),  # AI with high confidence
            ("Segment 5", 0, 0.9),  # Human with high confidence
        ]
        
        transitions = self.detector.detect_transitions(segment_results)
        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0]['from'], 'Human')
        self.assertEqual(transitions[0]['to'], 'AI')
        self.assertEqual(transitions[1]['from'], 'AI')
        self.assertEqual(transitions[1]['to'], 'Human')

if __name__ == "__main__":
    unittest.main()
