# main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import os

app = FastAPI(
    title="Mred GPT Project",
    description="Lightweight moderation GPT API ready for Railway deployment",
    version="1.0.0"
)

# Request schema
class ModerationRequest(BaseModel):
    text: str

# Lazy-load model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading model...")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            problem_type="multi_label_classification"
        )
        model.eval()
        print("Model loaded.")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/moderate")
async def moderate(request: ModerationRequest):
    load_model()  # Ensure model is loaded

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

    return {
        "text": request.text,
        "moderation_scores": response
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
