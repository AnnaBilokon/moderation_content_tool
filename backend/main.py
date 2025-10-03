import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://moderation-content-tool.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "typeform/distilbert-base-uncased-mnli"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


class ClassifyRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    multi_label: bool = True


@app.get("/health")
def health():
    if not HF_TOKEN:

        return {"status": "ok", "warning": "HF_TOKEN not set"}
    return {"status": "ok"}


def call_hf(text: str, labels: List[str], multi_label: bool, retries: int = 3):
    payload = {"inputs": text, "parameters": {
        "candidate_labels": labels, "multi_label": multi_label}}
    last_text = None
    for i in range(retries):
        r = requests.post(HF_API_URL, headers=HEADERS,
                          json=payload, timeout=30)
        last_text = r.text
        if r.status_code == 200:
            return r.json()

        if r.status_code in (503, 524) or "loading" in last_text.lower():
            time.sleep(min(1 + i, 3))
            continue

        raise HTTPException(status_code=r.status_code, detail=last_text)
    raise HTTPException(
        status_code=503, detail=last_text or "Hugging Face API unavailable")


@app.post("/classify")
def classify(req: ClassifyRequest):
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500, detail="HF_TOKEN is not configured on the server")

    texts = [t.strip() for t in req.texts if t and t.strip()]
    labels = [l.strip() for l in req.labels if l and l.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    if not labels:
        raise HTTPException(status_code=400, detail="No labels provided")

    results = []
    for text in texts[:128]:
        data = call_hf(text, labels, req.multi_label)

        pairs = [{"label": lab, "score": float(scr)} for lab, scr in zip(
            data["labels"], data["scores"])]

        picked = [p for p in pairs if p["score"] >=
                  req.threshold] if req.threshold else [pairs[0]] if pairs else []
        results.append({"text": text, "picked": picked, "all": pairs})
    return {"results": results}
