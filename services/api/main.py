from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_topic

app = FastAPI(title="Trustpilot Topic API")


class ReviewRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(review: ReviewRequest):
    return predict_topic(review.text)