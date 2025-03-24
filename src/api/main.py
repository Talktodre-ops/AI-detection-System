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
    # Append to feedback CSV
    df = pd.DataFrame([feedback.dict()])
    df.to_csv("datasets/feedback.csv", mode="a", header=False, index=False)
    return {"message": "Feedback recorded successfully!"}

# Ensure feedback.csv exists
if not os.path.exists("datasets/feedback.csv"):
    pd.DataFrame(columns=["text", "true_label"]).to_csv("datasets/feedback.csv", index=False)