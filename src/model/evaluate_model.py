import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import torch
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path)
        # Ensure no missing values
        self.data = self.data.dropna(subset=['cleaned_text'])
        # Ensure all text is string type
        self.data['cleaned_text'] = self.data['cleaned_text'].astype(str)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['cleaned_text']
        label = self.data.iloc[idx]['label']
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1 metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics

def evaluate_model(test_path, model_dir, max_length=64):
    # Check if the file exists
    if not os.path.exists(test_path):
        logging.error(f"Test file not found: {test_path}")
        return None
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model = model.to(device)
    
    # Create dataset
    test_dataset = TextDataset(test_path, tokenizer, max_length)
    logging.info(f"Test dataset loaded with {len(test_dataset)} samples")
    
    # Create trainer
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=32,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )
    
    # Evaluate
    logging.info("Starting evaluation...")
    raw_predictions = trainer.predict(test_dataset)
    predictions = raw_predictions.predictions.argmax(-1)
    labels = [int(label) for label in test_dataset.data["label"].values]
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions)
    
    logging.info(f"Test Results:")
    for metric_name, metric_value in metrics.items():
        logging.info(f"{metric_name.capitalize()}: {metric_value:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Use the cleaned test dataset if it exists
    test_path = "../data/datasets/test_cleaned.csv"
    if not os.path.exists(test_path):
        test_path = "../data/datasets/test.csv"
        logging.warning(f"Cleaned test dataset not found, using original: {test_path}")
    
    model_dir = "./models/distilroberta"
    evaluate_model(test_path, model_dir)
