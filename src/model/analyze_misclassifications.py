import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from lime import lime_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

def predict_proba(texts, tokenizer, model, device):
    """Return probabilities for all texts."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    return probs

def analyze_misclassifications(misclassified_path, model_dir, output_dir="./analysis/misclassifications"):
    """Analyze misclassified examples in detail using LIME."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load misclassified examples
    df = pd.read_csv(misclassified_path)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    
    # Create LIME explainer
    explainer = lime_text.LimeTextExplainer(
        class_names=["Human", "AI"],
        split_by="sentence"
    )
    
    # Analyze top 10 most confident misclassifications
    top_misclassified = df.sort_values('confidence', ascending=False).head(10)
    
    # Create HTML report
    html_report = """
    <html>
    <head>
        <title>Misclassification Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .example { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            .header { font-weight: bold; margin-bottom: 10px; }
            .text { margin-bottom: 20px; white-space: pre-wrap; }
            .explanation { margin-top: 20px; }
            .human { color: green; }
            .ai { color: red; }
        </style>
    </head>
    <body>
        <h1>Analysis of Misclassified Examples</h1>
    """
    
    # Function to wrap predict_proba for LIME
    def predict_proba_fn(texts):
        return predict_proba(texts, tokenizer, model, device)
    
    # Analyze each example
    for i, (_, row) in enumerate(top_misclassified.iterrows()):
        logging.info(f"Analyzing example {i+1}/10")
        
        text = row['cleaned_text']
        true_label = int(row['label'])
        pred_label = int(row['prediction'])
        
        # Get LIME explanation
        exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
        
        # Extract important features
        feature_importance = exp.as_list()
        
        # Add to HTML report
        html_report += f"""
        <div class="example">
            <div class="header">
                Example {i+1}: 
                True: <span class="{'human' if true_label == 0 else 'ai'}">{"Human" if true_label == 0 else "AI"}</span>, 
                Predicted: <span class="{'human' if pred_label == 0 else 'ai'}">{"Human" if pred_label == 0 else "AI"}</span>, 
                Confidence: {row['confidence']:.4f}
            </div>
            <div class="text">{text[:1000]}...</div>
            <div class="explanation">
                <h3>Key Influencing Factors:</h3>
                {exp.as_html()}
            </div>
        </div>
        """
        
        # Save individual explanation
        with open(os.path.join(output_dir, f"example_{i+1}_explanation.html"), 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Example {i+1} Explanation</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ font-weight: bold; margin-bottom: 10px; }}
                    .text {{ margin-bottom: 20px; white-space: pre-wrap; }}
                    .explanation {{ margin-top: 20px; }}
                    .human {{ color: green; }}
                    .ai {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Example {i+1} Detailed Analysis</h1>
                <div class="header">
                    True: <span class="{'human' if true_label == 0 else 'ai'}">{"Human" if true_label == 0 else "AI"}</span>, 
                    Predicted: <span class="{'human' if pred_label == 0 else 'ai'}">{"Human" if pred_label == 0 else "AI"}</span>, 
                    Confidence: {row['confidence']:.4f}
                </div>
                <div class="text">{text}</div>
                <div class="explanation">
                    <h3>Key Influencing Factors:</h3>
                    {exp.as_html()}
                </div>
            </body>
            </html>
            """)
    
    # Complete HTML report
    html_report += """
    </body>
    </html>
    """
    
    # Save HTML report
    with open(os.path.join(output_dir, "misclassification_analysis.html"), 'w') as f:
        f.write(html_report)
    
    logging.info(f"Analysis complete. Report saved to {os.path.join(output_dir, 'misclassification_analysis.html')}")
    
    # Analyze patterns in misclassifications
    human_as_ai = df[df['label'] == 0]
    ai_as_human = df[df['label'] == 1]
    
    # Word count analysis
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(re.findall(r'\w+', x)))
    
    avg_word_count = {
        'human_as_ai': human_as_ai['word_count'].mean(),
        'ai_as_human': ai_as_human['word_count'].mean()
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, "misclassification_stats.txt"), 'w') as f:
        f.write("Misclassification Statistics\n")
        f.write("===========================\n\n")
        f.write(f"Total misclassifications: {len(df)}\n")
        f.write(f"Human text classified as AI: {len(human_as_ai)}\n")
        f.write(f"AI text classified as human: {len(ai_as_human)}\n\n")
        f.write("Average word count:\n")
        f.write(f"  Human text classified as AI: {avg_word_count['human_as_ai']:.1f} words\n")
        f.write(f"  AI text classified as human: {avg_word_count['ai_as_human']:.1f} words\n")
    
    logging.info(f"Statistics saved to {os.path.join(output_dir, 'misclassification_stats.txt')}")
    
    return {
        "total_misclassified": len(df),
        "human_as_ai": len(human_as_ai),
        "ai_as_human": len(ai_as_human),
        "avg_word_count": avg_word_count
    }

if __name__ == "__main__":
    misclassified_path = "./analysis/misclassified_examples.csv"
    model_dir = "./models/distilroberta"
    
    if not os.path.exists(misclassified_path):
        logging.error(f"Misclassified examples file not found: {misclassified_path}")
        logging.info("Please run analyze_results.py first to generate the misclassified examples file.")
    else:
        analyze_misclassifications(misclassified_path, model_dir)