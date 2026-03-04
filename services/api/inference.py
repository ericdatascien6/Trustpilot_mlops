import os
import joblib
import numpy as np
import re
import mlflow.sklearn
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_NAME = "TrustpilotTopicModel"
MODEL_STAGE = "Production"

#kmeans = joblib.load(MODELS_DIR / "kmeans_topics.pkl")
model = mlflow.sklearn.load_model(
    model_uri=f"models:/{MODEL_NAME}@{MODEL_STAGE}"
)

cluster_labels = joblib.load(MODELS_DIR / "cluster_labels.pkl")

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def clean_review_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_topic(review_text: str):

    review_text = clean_review_text(review_text)

    if not review_text.strip():
        return {
            "cluster_id": -1,
            "theme": "Empty input",
            "confidence": 0.0
        }

    embedding = sbert_model.encode([review_text])
    embedding = np.asarray(embedding)

    #cluster_id = int(kmeans.predict(embedding)[0])
    cluster_id = int(model.predict(embedding)[0])

    distances = model.transform(embedding)
    min_distance = float(np.min(distances))

    theme = cluster_labels.get(cluster_id, "Unknown")

    return {"cluster_id": cluster_id, "theme": theme, "confidence": min_distance}
