import os
import re
from typing import List, Optional, Literal, Dict, Any

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_ID = "leolee99/PIGuard"

Stage = Literal["user_input", "rag_chunk", "tool_output", "action_plan"]
Label = Literal["benign", "instruction_hijack", "data_exfiltration", "tool_misuse", "prompt_leak", "other_malicious"]

# -------- Stage-specific thresholds (tune later) --------
# RAG/tool boundaries: keep LOW FPR (don’t over-block)
THRESH = {
    "user_input": 0.85,
    "rag_chunk": 0.95,
    "tool_output": 0.95,
    # action_plan boundary: keep LOW FNR (block more aggressively)
    "action_plan": 0.70,
}

# -------- High-precision type rules (start here, extend later) --------
RE_PROMPT_LEAK = re.compile(r"(system prompt|developer message|reveal.*instructions|show.*policy|hidden prompt)", re.I)
RE_HIJACK = re.compile(r"(ignore (all|previous) instructions|override|disregard|you must follow|system:|developer:)", re.I)
RE_EXFIL = re.compile(r"(api key|token|password|secret|private|conversation history|chat history|send to|upload to|exfil)", re.I)
RE_TOOL = re.compile(r"(run (this )?(command|shell)|execute|powershell|bash|curl |wget |pip install|rm -rf|chmod|ssh |nc |netcat)", re.I)

def classify_type(text: str) -> Label:
    """
    Rule-based multi-class labeling for injection *intent*.
    Only used if PIGuard says text is likely injection.
    """
    t = text.strip()
    if RE_PROMPT_LEAK.search(t):
        return "prompt_leak"
    if RE_EXFIL.search(t):
        return "data_exfiltration"
    if RE_TOOL.search(t):
        return "tool_misuse"
    if RE_HIJACK.search(t):
        return "instruction_hijack"
    return "other_malicious"

def auth_or_401(x_api_key: Optional[str]):
    pass

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
    max_length=1048,
    device=DEVICE,
)

# -------- API schema --------
app = FastAPI(title="PIGuard Multi-class API", version="1.0")

class DetectRequest(BaseModel):
    texts: List[str]
    stage: Stage = "rag_chunk"

class DetectItem(BaseModel):
    label: Label
    is_injection: bool
    score: float
    blocked: bool
    action: Literal["pass", "sanitize", "block"]
    meta: Optional[Dict[str, Any]] = None

class DetectResponse(BaseModel):
    model: str
    stage: Stage
    results: List[DetectItem]

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": "cuda" if DEVICE == 0 else "cpu"}

@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest):

    if not req.texts or len(req.texts) > 128:
        raise HTTPException(status_code=400, detail="texts must be 1..128 items")
    if any(len(t) > 20000 for t in req.texts):
        raise HTTPException(status_code=400, detail="text too long")

    stage = req.stage
    thresh = THRESH.get(stage, 0.90)

    with torch.no_grad():
        out = clf(req.texts)

    results: List[DetectItem] = []
    for text, o in zip(req.texts, out):
        # o = {"label": "...", "score": ...}
        base_label = str(o["label"])
        score = float(o["score"])

        # PIGuard typically outputs something like benign/injection (depends on label2id)
        # We infer injection if label != benign OR score above threshold when label indicates injection.
        # If your model uses different label strings, adjust this function.
        is_benign = ("benign" in base_label.lower())
        is_injection = (not is_benign) and (score >= thresh)

        if not is_injection:
            results.append(DetectItem(
                label="benign",
                is_injection=False,
                score=score,
                blocked=False,
                action="pass",
                meta={"raw_label": base_label, "threshold": thresh},
            ))
            continue

        inj_type = classify_type(text)

        # action policy: at rag/tool boundary -> sanitize; at action_plan -> block
        if stage == "action_plan":
            action = "block"
            blocked = True
        else:
            action = "sanitize"
            blocked = True

        results.append(DetectItem(
            label=inj_type,
            is_injection=True,
            score=score,
            blocked=blocked,
            action=action,
            meta={"raw_label": base_label, "threshold": thresh},
        ))

    return DetectResponse(model=MODEL_ID, stage=stage, results=results)
