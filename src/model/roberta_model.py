import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch
import logging

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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to("cuda")

    # Use absolute paths for datasets (adjust as needed)
    train_path = r"C:\Users\talkt\Documents\GitHub\AI-generated-content-detection\src\data\datasets\train.csv"
    val_path = r"C:\Users\talkt\Documents\GitHub\AI-generated-content-detection\src\data\datasets\val.csv"

    train_dataset = TextDataset(train_path, tokenizer, max_length=64)
    val_dataset = TextDataset(val_path, tokenizer, max_length=64)

    training_args = TrainingArguments(
        output_dir="./model/models/distilroberta",  # Ensure this matches your directory structure
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        fp16=True,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",  # Fixed deprecated warning
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained("./model/models/distilroberta")
    tokenizer.save_pretrained("./model/models/distilroberta")
    logging.info("Model and tokenizer saved to ./model/models/distilroberta")

if __name__ == "__main__":
    train_model()