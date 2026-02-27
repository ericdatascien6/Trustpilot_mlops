import os
import re
import joblib
import pandas as pd

import mlflow
from mlflow import log_params, log_metric, log_artifacts

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DATA_PATH = os.getenv("DATA_PATH", "/data/raw/train.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot-topic-modeling")

K = int(os.getenv("K", "6"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20000"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

def clean_for_sbert(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH, header=None, names=["label", "title", "text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    df["text_sbert"] = df["text"].apply(clean_for_sbert)

    df_sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

    with mlflow.start_run(run_name="sbert_kmeans_training"):
        log_params({
            "k": K,
            "sample_size": len(df_sample),
            "random_state": RANDOM_STATE,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "algo": "KMeans"
        })

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedder.encode(df_sample["text_sbert"].tolist(), show_progress_bar=True)

        kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        sil = silhouette_score(embeddings, labels)
        log_metric("silhouette", float(sil))

        # Export artefacts compatibles avec le  service d'inference
        joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_topics.pkl"))
        joblib.dump({i: f"Theme_{i}" for i in range(K)}, os.path.join(MODEL_DIR, "cluster_labels.pkl"))

        # Log artefacts dans MLflow
        log_artifacts(MODEL_DIR, artifact_path="exported_models")

        print(f"✅ Training terminé. Silhouette={sil:.4f}")
        print(f"✅ Modèles exportés dans {MODEL_DIR}")

if __name__ == "__main__":
    main()
