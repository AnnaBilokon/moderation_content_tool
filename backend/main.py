from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from functools import lru_cache
from transformers import pipeline
import os

app = FastAPI(title="Content Moderation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    texts: List[str]
    labels: List[str] = Field(default_factory=lambda: [
        "toxic", "insult", "harassment", "hate_speech", "racism", "sexism",
        "sexual_content", "self_harm", "spam", "safe"
    ])
    multi_label: bool = True
    threshold: float = Field(default=float(
        os.getenv("DEFAULT_THRESHOLD", 0.7)), ge=0.0, le=1.0)


class LabelScore(BaseModel):
    label: str
    score: float


class ClassifyItem(BaseModel):
    text: str
    picked: List[LabelScore]
    all: List[LabelScore]


class ClassifyResponse(BaseModel):
    results: List[ClassifyItem]


@lru_cache(maxsize=1)
def get_clf():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    texts = [t.strip() for t in req.texts if t and t.strip()]
    labels = [l.strip() for l in req.labels if l and l.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided.")
    if not labels:
        raise HTTPException(status_code=400, detail="No labels provided.")

    clf = get_clf()

    thr = float(req.threshold)
    results = []
    for t in texts[:500]:
        res = clf(t, candidate_labels=labels, multi_label=req.multi_label)
        pairs = [{"label": lab, "score": float(scr)} for lab, scr in zip(
            res["labels"], res["scores"])]
        picked = [p for p in pairs if p["score"] >= thr] or (
            [pairs[0]] if pairs else [])
        results.append({"text": t, "picked": picked, "all": pairs})
    return {"results": results}
