from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nest_asyncio
from pyngrok import ngrok, conf
import uvicorn
import threading
import time

# Allow nested loops for Colab / environments
nest_asyncio.apply()

# API Key (optional for user-facing if you want restriction, for now public test no restriction)
API_KEY = None  # Leave None for public access

# FastAPI initialization
app = FastAPI(
    title="Mred GPT API",
    description="FastAPI for Unfiltered ChatGPT simulation with Transformer model",
    version="1.0.0"
)

# Model setup
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

# Optional: API Key middleware
@app.middleware("http")
async def api_key_check(request: Request, call_next):
    if API_KEY:
        if request.headers.get("x-api-key") != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    response = await call_next(request)
    return response

# Main endpoint
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

    # Log entry
    log_entry = f"Input: {request.text}\nScores: {response}\n\n"
    with open("moderation_log.txt", "a") as log_file:
        log_file.write(log_entry)

    return {
        "text": request.text,
        "moderation_scores": response,
        "log": "Saved to moderation_log.txt"
    }

# Ngrok auto-start
def start_ngrok():
    conf.get_default().auth_token = "2vMeQOOPc8Rki7ISlvbVZljJ9Bl_5MNXebAh8rugW4RzfRJQJ"  # your token
    max_retries = 5
    for attempt in range(max_retries):
        try:
            public_url = ngrok.connect(8000)
            print(f"Public API is live at: {public_url}")
            break
        except Exception as e:
            print(f"Ngrok error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
    else:
        print("Ngrok tunnel failed after multiple attempts.")

# Server starter
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run threads
threading.Thread(target=start_server).start()
threading.Thread(target=start_ngrok).start()
