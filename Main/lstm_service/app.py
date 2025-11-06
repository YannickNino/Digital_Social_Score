
import os, json
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from preprocess import clean_text

MAX_LEN = 120
MODEL_PATH = "model.keras"

with open("tokenizer.json", "r", encoding="utf-8") as f:
    tok_json = f.read()
tokenizer = tokenizer_from_json(tok_json)

with open("labels.txt", "r", encoding="utf-8") as f:
    LABELS = [line.strip() for line in f if line.strip()]

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="Toxic Comment LSTM API", version="1.0")

class PredictIn(BaseModel):
    texts: List[str]

class PredictOut(BaseModel):
    scores: List[Dict[str, float]]

@app.get("/health")
def health():
    return {"status": "ok", "labels": LABELS}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    cleaned = [clean_text(t) for t in payload.texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    pad = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    preds = model.predict(pad, verbose=0)
    out = []
    for row in preds.tolist():
        out.append({LABELS[i]: float(row[i]) for i in range(len(LABELS))})
    return PredictOut(scores=out)
