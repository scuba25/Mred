from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

# FastAPI app
app = FastAPI(
    title="Moderation API",
    description="Public API for unrestricted moderation chat",
    version="1.0.0"
)

API_KEY = "2vMeQOOPc8Rki7ISlvbVZljJ9Bl_5MNXebAh8rugW4RzfRJQJ"

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)
model.eval()

class ModerationRequest(BaseModel):
    text: str

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid API Key.")
    return await call_next(request)

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

    return {
        "text": request.text,
        "moderation_scores": response
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
  
