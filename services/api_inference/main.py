from fastapi import FastAPI
from pydantic import BaseModel
from services.api_inference.inference import predict_topic

app = FastAPI(title="Trustpilot Topic API")


class ReviewRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(review: ReviewRequest):
    return predict_topic(review.text)


@app.get("/health")
def health():
    return {"status": "ok"}
