import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix, classification_report
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

def analyze_model_results(test_path, model_dir, max_length=512, output_dir="./analysis"):
    """Analyze model results with confusion matrix and misclassification examples."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    df_test = pd.read_csv(test_path)
    df_test = df_test.dropna(subset=['cleaned_text'])
    df_test['cleaned_text'] = df_test['cleaned_text'].astype(str)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    
    # Get predictions
    all_predictions = []
    all_probs = []
    batch_size = 32
    
    for i in range(0, len(df_test), batch_size):
        batch_texts = df_test['cleaned_text'].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
        all_predictions.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        if i % 1000 == 0:
            logging.info(f"Processed {i}/{len(df_test)} examples")
    
    # Add predictions to dataframe
    df_test['prediction'] = all_predictions
    df_test['confidence'] = [prob[pred] for prob, pred in zip(all_probs, all_predictions)]
    
    # Calculate metrics
    true_labels = df_test['label'].astype(int).values
    pred_labels = df_test['prediction'].values
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    logging.info(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    # Classification report
    report = classification_report(true_labels, pred_labels, target_names=['Human', 'AI'])
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    logging.info(f"Classification report saved to {os.path.join(output_dir, 'classification_report.txt')}")
    
    # Find misclassified examples
    df_misclassified = df_test[df_test['label'] != df_test['prediction']]
    
    # Sort by confidence (most confident mistakes first)
    df_misclassified = df_misclassified.sort_values('confidence', ascending=False)
    
    # Save top misclassified examples
    misclassified_path = os.path.join(output_dir, 'misclassified_examples.csv')
    df_misclassified.to_csv(misclassified_path, index=False)
    logging.info(f"Misclassified examples saved to {misclassified_path}")
    
    # Print summary of misclassifications
    human_as_ai = ((df_test['label'] == 0) & (df_test['prediction'] == 1)).sum()
    ai_as_human = ((df_test['label'] == 1) & (df_test['prediction'] == 0)).sum()
    
    logging.info(f"Total misclassifications: {len(df_misclassified)}")
    logging.info(f"Human text classified as AI: {human_as_ai}")
    logging.info(f"AI text classified as human: {ai_as_human}")
    
    # Sample of misclassified examples (first 5)
    logging.info("\nSample of misclassified examples:")
    for i, (_, row) in enumerate(df_misclassified.head(5).iterrows()):
        true_label = "Human" if row['label'] == 0 else "AI"
        pred_label = "Human" if row['prediction'] == 0 else "AI"
        logging.info(f"Example {i+1}:")
        logging.info(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {row['confidence']:.4f}")
        logging.info(f"  Text: {row['cleaned_text'][:200]}...")
        logging.info("-" * 80)
    
    return {
        "confusion_matrix": cm,
        "misclassified_count": len(df_misclassified),
        "human_as_ai": human_as_ai,
        "ai_as_human": ai_as_human
    }

if __name__ == "__main__":
    test_path = "../data/datasets/test_cleaned.csv"
    if not os.path.exists(test_path):
        test_path = "../data/datasets/test.csv"
        logging.warning(f"Cleaned test dataset not found, using original: {test_path}")
    
    model_dir = "./models/distilroberta"
    analyze_model_results(test_path, model_dir)