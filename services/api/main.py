import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from inference import predict_topic
#from training import train_model

API_KEY = os.getenv("API_KEY", "secret123")

app = FastAPI(title="Trustpilot Topic API")

Instrumentator().instrument(app).expose(app)

prediction_counter = Counter(
    "trustpilot_predictions_total",
    "Total number of predictions made by the API"
)

cluster_counter = Counter(
    "trustpilot_cluster_predictions_total",
    "Number of predictions per cluster",
    ["cluster_id"]
)

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
    result = predict_topic(review.text)
    prediction_counter.inc()
    cluster_counter.labels(cluster_id=str(result["cluster_id"])).inc()
    return predict_topic(review.text)


@app.get("/health")
def health():
    return {"status": "ok"}
