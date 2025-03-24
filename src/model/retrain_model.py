import os
import sys
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
from src.data.preprocessing import clean_text  # For text cleaning
from src.data.split_data import split_dataset  # For splitting data

sys.path.append('src/data')

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
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def main():
    # Load original cleaned data
    original_data = pd.read_csv("src/data/datasets/cleaned_AI_Human.csv")
    
    # Load user submissions (drop confidence column)
    user_submissions = pd.read_csv("src/data/datasets/user_submissions.csv")
    user_data = user_submissions[["text", "predicted_label"]].copy()
    
    # Clean user text and rename columns
    user_data["cleaned_text"] = user_data["text"].apply(clean_text)
    user_data = user_data.rename(columns={"predicted_label": "label"})
    user_data = user_data[["cleaned_text", "label"]].dropna()
    
    # Combine datasets
    combined_data = pd.concat([original_data, user_data]).drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    
    # Split combined data
    split_dir = "src/data/datasets/"
    split_dataset(
        data=combined_data,
        train_path=os.path.join(split_dir, "retrained_train.csv"),
        val_path=os.path.join(split_dir, "retrained_val.csv"),
        test_path=os.path.join(split_dir, "retrained_test.csv"),
    )
    
    # Retrain the model
    model_dir = "src/model/models/distilroberta"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda")
    
    train_dataset = TextDataset(
        data_path=os.path.join(split_dir, "retrained_train.csv"),
        tokenizer=tokenizer,
        max_length=64,
    )
    val_dataset = TextDataset(
        data_path=os.path.join(split_dir, "retrained_val.csv"),
        tokenizer=tokenizer,
        max_length=64,
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,  # Increase epochs for better learning
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Model retrained with new user data!")

if __name__ == "__main__":
    main()