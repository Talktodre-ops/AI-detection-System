from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

# Define feedback model
class Feedback(BaseModel):
    text: str
    true_label: int  # 0 for human, 1 for AI

@app.post("/flag_prediction/")
async def flag_prediction(feedback: Feedback):
    """
    Flag a prediction as incorrect and store it for retraining.
    """
    # Ensure the datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    
    # Append to feedback CSV
    df = pd.DataFrame([feedback.model_dump()])
    df.to_csv("datasets/feedback.csv", mode="a", header=False, index=False)
    return {"message": "Feedback recorded successfully!"}

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the AI Detection System API"}

@app.post("/predict/")
async def predict(text_input: dict):
    """
    Predict whether text is AI-generated or human-written.
    
    Args:
        text_input (dict): A dictionary containing the text to analyze
        
    Returns:
        dict: Prediction results with label and probability
    """
    text = text_input.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    # Mock prediction for testing
    import random
    prediction = random.randint(0, 1)
    probability = random.random()
    
    return {
        "prediction": prediction,  # 0 for human, 1 for AI
        "probability": probability
    }
