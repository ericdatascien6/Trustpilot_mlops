import joblib
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

kmeans = joblib.load(MODELS_DIR / "kmeans_topics.pkl")
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

    cluster_id = int(kmeans.predict(embedding)[0])

    distances = kmeans.transform(embedding)
    min_distance = float(np.min(distances))

    theme = cluster_labels.get(cluster_id, "Unknown")

    return {"cluster_id": cluster_id, "theme": theme, "confidence": min_distance}
