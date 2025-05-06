import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments  # Changed import statement
from torch.utils.data import Dataset
import torch
import logging
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.config import TRAIN_DATASET, VAL_DATASET, MODEL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path).dropna()
        self.data = self.data[self.data['cleaned_text'].str.strip() != ""]
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

def train_model():
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)

    # Ensure the datasets exist
    for dataset_path in [TRAIN_DATASET, VAL_DATASET]:
        if not os.path.exists(dataset_path):
            logging.error(f"Dataset not found: {dataset_path}")
            logging.info("Please run data/preprocess_data.py and data/split_data.py first.")
            return False

    train_dataset = TextDataset(TRAIN_DATASET, tokenizer, max_length=64)
    val_dataset = TextDataset(VAL_DATASET, tokenizer, max_length=64)
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Use only basic parameters that should work with any version
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        # Remove problematic parameters
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    logging.info(f"Model and tokenizer saved to {MODEL_DIR}")
    return True

if __name__ == "__main__":
    success = train_model()
    
    if success:
        logging.info("Model training completed successfully.")
    else:
        logging.info("Model training failed.")
