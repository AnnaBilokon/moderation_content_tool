import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "typeform/distilbert-base-uncased-mnli"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


class ClassifyRequest(BaseModel):
    texts: List[str]
    labels: List[str]
    multi_label: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
def classify(req: ClassifyRequest):
    results = []
    for text in req.texts:
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": req.labels,
                "multi_label": req.multi_label,
            },
        }
        r = requests.post(HF_API_URL, headers=HEADERS,
                          json=payload, timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()

        pairs = [{"label": lab, "score": float(scr)} for lab, scr in zip(
            data["labels"], data["scores"])]
        results.append({"text": text, "picked": [pairs[0]], "all": pairs})
    return {"results": results}
