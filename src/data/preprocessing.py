import re
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
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
    return tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")