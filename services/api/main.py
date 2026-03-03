import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

from inference import predict_topic
#from training import train_model

API_KEY = os.getenv("API_KEY", "secret123")

app = FastAPI(title="Trustpilot Topic API")

# Authentification simple
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ==============================
# Schéma requête inference
# ==============================

class ReviewRequest(BaseModel):
    text: str


# ==============================
# Routes API
# ==============================

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(review: ReviewRequest):
    return predict_topic(review.text)


@app.get("/health")
def health():
    return {"status": "ok"}
