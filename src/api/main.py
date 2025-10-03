from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from src.api.fact_checker import verify_claims
# from src.api.database import save_feedback  # Keep this commented if not needed

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8081",  # Added for alternate port
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "models", "distilroberta")

tokenizer = None
model = None
device = None
torch = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_DIR}: {e}")
    print("API will run in mock mode for testing")

# Define feedback model
class Feedback(BaseModel):
    text: str
    true_label: int  # 0 for human, 1 for AI

class TextIn(BaseModel):
    text: str

class MixedContentRequest(BaseModel):
    text: str
    segment_type: str = "sentence"  # "sentence" or "paragraph"
    threshold: float = 0.7

@app.post("/fact-check")
@app.post("/fact-check/")
async def fact_check(text_in: TextIn):
    """Verify factual claims in the provided text."""
    return await verify_claims(text_in.text)

# @app.post("/flag_prediction/")
# async def flag_prediction(feedback: Feedback):
#     """Flag a prediction as incorrect and store it for retraining."""
#     success = save_feedback(feedback.model_dump())
#     if success:
#         return {"message": "Feedback recorded successfully!"}
#     else:
#         raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the AI Detection System API"}

@app.post("/analyze-mixed-content/")
async def analyze_mixed_content(request: MixedContentRequest):
    """
    Analyze text for mixed AI and human content.
    Returns segment-level analysis with transitions.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input is required")

    try:
        from src.model.mixed_content_detector import MixedContentDetector

        # Initialize detector with current model
        detector = MixedContentDetector(
            model_dir=MODEL_DIR,
            device=device,
            segment_type=request.segment_type
        )

        # Analyze the text
        results = detector.analyze_text(request.text, threshold=request.threshold)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
        # Mock mode for testing when model is not available
        import random
        prediction = random.choice([0, 1])
        probability = random.uniform(0.7, 0.95)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": "AI-generated" if prediction == 1 else "Human-written"
        }

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
