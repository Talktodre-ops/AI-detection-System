import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path)
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

def evaluate_model(test_path, model_dir, max_length=64):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda")
    
    test_dataset = TextDataset(test_path, tokenizer, max_length)
    
    trainer = Trainer(model=model)
    raw_predictions = trainer.predict(test_dataset)
    predictions = raw_predictions.predictions.argmax(-1)
    labels = test_dataset.data["label"].values
    #model calculation 
    accuracy = (predictions == labels).mean()
    precision = precision_score(labels, predictions, average= 'binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average= 'binary')

    logging.info(f"Test Results:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"f1: {f1:.4f}")


if __name__ == "__main__":
    test_path = "../data/datasets/test.csv"
    model_dir = "./models/distilroberta"
    evaluate_model(test_path, model_dir)