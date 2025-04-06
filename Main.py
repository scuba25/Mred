# Install dependencies manually in requirements.txt
# fastapi
# uvicorn
# transformers
# torch
# nest-asyncio

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nest_asyncio
import os

# Apply nested asyncio for environments like Colab (safe to keep)
nest_asyncio.apply()

# Optional: Simple API Key for protection (can remove or disable later)
API_KEY = "2vMeQOOPc8Rki7ISlvbVZljJ9Bl_5MNXebAh8rugW4RzfRJQJ"  # <-- your actual key

# Initialize FastAPI app
app = FastAPI(
    title="Mred GPT API",
    description="Custom GPT-style API for unrestricted text moderation or other tasks.",
    version="1.0.0"
)

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3  # Update if needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)
model.eval()

# Define request schema
class ModerationRequest(BaseModel):
    text: str

# Middleware for API Key checking (optional)
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid API Key.")
    return await call_next(request)

# Main moderation endpoint
@app.post("/moderate")
async def moderate(request: ModerationRequest):
    # Tokenize input
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    # Define labels
    labels = ["Toxic", "Violent", "Other"]
    response = {label: round(prob, 3) for label, prob in zip(labels, probs)}

    # Logging (saved to local file)
    log_entry = f"Input: {request.text}\nScores: {response}\n\n"
    with open("moderation_log.txt", "a") as log_file:
        log_file.write(log_entry)

    return {
        "text": request.text,
        "moderation_scores": response,
        "log": "Saved to moderation_log.txt"
    }

# Get port from environment for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
