from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.api.fact_checker import verify_claims
from src.api.database import save_feedback

app = FastAPI()

# Load model at startup
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "models", "distilroberta")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_DIR}: {e}")
    tokenizer = None
    model = None

# Define feedback model
class Feedback(BaseModel):
    text: str
    true_label: int  # 0 for human, 1 for AI

class TextIn(BaseModel):
    text: str

@app.post("/fact-check")
@app.post("/fact-check/")
async def fact_check(text_in: TextIn):
    """Verify factual claims in the provided text."""
    return await verify_claims(text_in.text)

@app.post("/flag_prediction/")
async def flag_prediction(feedback: Feedback):
    """Flag a prediction as incorrect and store it for retraining."""
    success = save_feedback(feedback.model_dump())
    if success:
        return {"message": "Feedback recorded successfully!"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the AI Detection System API"}

@app.post("/predict/")
async def predict(text_input: TextIn):
    """
    Predict whether text is AI-generated or human-written.

    Args:
        text_input: TextIn model containing the text to analyze

    Returns:
        dict: Prediction results with label and probability
    """
    if not text_input.text or not text_input.text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # Tokenize and predict
    inputs = tokenizer(
        text_input.text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        probability = probs[0][prediction].item()

    return {
        "prediction": prediction,  # 0 for human, 1 for AI
        "probability": round(probability, 4),
        "label": "AI-generated" if prediction == 1 else "Human-written"
    }
