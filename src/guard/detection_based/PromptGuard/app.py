from typing import List, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Available models:
# Llama Prompt Guard 2 86M	81.2%
# Llama Prompt Guard 2 22M	78.4%

# MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-22M"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
Label = Literal["benign", "prompt_injection"] # benign, malicious
RawLabel = Literal["LABEL_0", "LABEL_1"] # benign, malicious

# -------- Load PIGuard --------
DEVICE = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# We want scores we can threshold: pipeline returns label + score
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=DEVICE,
)

# -------- API schema --------
app = FastAPI(title="PromptGuard Multi-class API", version="1.0")

class DetectRequest(BaseModel):
    texts: List[str]

class DetectItem(BaseModel):
    label: Label
    confidence: float

class DetectResponse(BaseModel):
    model: str
    results: List[DetectItem]


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": "cuda" if DEVICE == 0 else "cpu"}

@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest):

    if not req.texts or len(req.texts) > 128:
        raise HTTPException(status_code=400, detail="texts must be 1..128 items")
    texts = [text[:20000] for text in req.texts]

    with torch.no_grad():
        out = clf(req.texts)

    results: List[DetectItem] = []
    for o in out:
        raw_label = o['label']
        conf = float(o["score"])
        print(raw_label, conf)

        if raw_label == "LABEL_0":
            results.append(DetectItem(
                label="benign",
                confidence=conf,
            ))
            continue

        results.append(DetectItem(
            label="prompt_injection",
            confidence=conf,
        ))

    return DetectResponse(model=MODEL_ID, results=results)


# uvicorn app:app --host 127.0.0.1 --port 12390