import re
import pandas as pd
import numpy as np
from transformers import BertTokenizer

def clean_text(text):
    """
    Clean the input text by converting it to lowercase and removing special characters.
    and collapsing white spaces

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    # Use the more comprehensive preprocess_text function
    return preprocess_text(text)

def tokenize_text(text, tokenizer, max_length=128):
    """
    Tokenize the input text using a pre-trained BERT tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum length of the tokenized sequence.

    Returns:
        dict: The tokenized input IDs and attention mask.
    """
    # First clean the text
    cleaned_text = preprocess_text(text)
    return tokenizer(cleaned_text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def preprocess_text(text):
    """
    Preprocess text for model input
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Handle NaN, None, or empty values
    if pd.isna(text) or text is None or text == "":
        return ""
        
    # Ensure text is a string
    text = str(text)
        
    # Basic preprocessing
    text = text.strip()
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
