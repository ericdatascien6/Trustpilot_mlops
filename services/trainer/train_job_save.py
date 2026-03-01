import os
import re
import pickle
import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


DATA_PATH = os.getenv("DATA_PATH", "/data/raw/train.csv")
MODELS_DIR = os.getenv("MODELS_DIR", "/models")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K = int(os.getenv("K", "6"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20000"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot_topic_modeling")


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"[^a-z0-9\s\.\,\!\?']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, header=None, names=["label", "title", "text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() > 0]

    if len(df) == 0:
        raise ValueError("Dataset vide après nettoyage")

    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="kmeans_sbert_training"):
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("k", K)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("sample_size", len(sample_df))

        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(sample_df["text"].tolist(), show_progress_bar=False)

        kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        sil = silhouette_score(embeddings, labels)
        mlflow.log_metric("silhouette", float(sil))

        kmeans_path = os.path.join(MODELS_DIR, "kmeans_topics.pkl")
        labels_path = os.path.join(MODELS_DIR, "cluster_labels.pkl")

        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)

        with open(labels_path, "wb") as f:
            pickle.dump(labels, f)

        mlflow.log_artifact(kmeans_path, artifact_path="model")
        mlflow.log_artifact(labels_path, artifact_path="model")

        print({"status": "success", "silhouette": float(sil), "k": K, "sample_size": len(sample_df)})


if __name__ == "__main__":
    main()
