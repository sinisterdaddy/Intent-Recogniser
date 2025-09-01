from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from langchain_chat.classifier import predict_intent  # We'll redefine below
from langchain_chat.dialogue_manager import generate_response, reset_conversation
from langchain_chat.entity_extractor import extract_entities

app = FastAPI(
    title="Intent Classifier API",
    description="Serving multiple NLP models",
    version="1.0"
)

# Load pipelines once at startup
distilbert_pipeline = pipeline("text-classification", model="./distilbert", tokenizer="./distilbert")
roberta_pipeline = pipeline("text-classification", model="./roberta", tokenizer="./roberta")
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/chat.html")

class Query(BaseModel):
    text: str
    model: str  # Expected: "distilbert", "roberta", "zero_shot"

def predict_intent(text: str, model_name: str):
    if model_name == "distilbert":
        preds = distilbert_pipeline(text, top_k=3)
    elif model_name == "roberta":
        preds = roberta_pipeline(text, top_k=3)
    elif model_name == "zero_shot":
        # Customize your candidate labels as needed
        candidate_labels = ["book_flight", "cancel_flight", "greeting", "weather", "play_music"]
        preds_raw = zero_shot_pipeline(text, candidate_labels=candidate_labels, multi_label=False)
        preds = [{"label": l, "score": s} for l, s in zip(preds_raw["labels"], preds_raw["scores"])]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Sort predictions descending by score
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)

    top_intent = preds[0]['label']
    confidence = preds[0]['score']

    return {
        "top_intent": top_intent,
        "confidence": confidence,
        "all_intents": preds
    }

import json
import re

@app.post("/chat")
def chat(query: Query):
    text = query.text
    model = query.model

    # Predict intent and confidence
    intent_result = predict_intent(text, model)
    top_intent = intent_result.get("top_intent")
    confidence = intent_result.get("confidence")
    all_intents = intent_result.get("all_intents")

    # Extract entities (make sure extract_entities returns dict)
    entities = extract_entities(text)

    # Generate response using top intent
    response = generate_response(intent=top_intent, entities=entities, user_input=text)

    return {
        "intent": top_intent,
        "confidence": confidence,
        # "all_intents": all_intents,
        # "entities": entities,
        "response": response
    }

@app.post("/reset")
def reset():
    reset_conversation()
    return {"message": "Conversation history cleared."}

@app.post("/predict/distilbert")
def predict_distilbert(query: Query):
    preds = distilbert_pipeline(query.text, top_k=3)
    return {"model": "distilbert", "input": query.text, "predictions": preds}

@app.post("/predict/roberta")
def predict_roberta(query: Query):
    preds = roberta_pipeline(query.text, top_k=3)
    return {"model": "roberta", "input": query.text, "predictions": preds}

class ZeroShotQuery(BaseModel):
    text: str
    candidate_labels: list[str]

@app.post("/predict/zero_shot")
def predict_zero_shot(query: ZeroShotQuery):
    preds = zero_shot_pipeline(query.text, candidate_labels=query.candidate_labels, multi_label=False)
    return {
        "model": "zero_shot",
        "input": query.text,
        "predictions": [
            {"label": label, "score": float(score)}
            for label, score in zip(preds['labels'], preds['scores'])
        ]
    }
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides the PORT env var
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
