# Install dependencies if needed (optional)
# pip install -r requirements.txt

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nest_asyncio
import uvicorn
import os

# Allow nested event loops
nest_asyncio.apply()

# Optional API Key protection
API_KEY = "your_secret_key_here"

# Init FastAPI
app = FastAPI(
    title="Moderation API",
    description="API for text moderation using a Transformer model.",
    version="1.0.0"
)

# Load model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)
model.eval()

# Request body schema
class ModerationRequest(BaseModel):
    text: str

# Middleware for API Key
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

# Main route
@app.post("/moderate")
async def moderate(request: ModerationRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    labels = ["Toxic", "Violent", "Other"]
    response = {label: round(prob, 3) for label, prob in zip(labels, probs)}

    # Logging
    with open("moderation_log.txt", "a") as log_file:
        log_file.write(f"Input: {request.text}\nScores: {response}\n\n")

    return {
        "text": request.text,
        "moderation_scores": response,
        "log": "Saved to moderation_log.txt"
    }

# Server runner
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
