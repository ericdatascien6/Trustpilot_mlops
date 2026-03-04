import os
import re
import time
import argparse
import joblib
import pandas as pd
import sklearn

import mlflow
import mlflow.sklearn
from mlflow import log_params, log_metric, log_artifacts

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DATA_PATH = os.getenv("DATA_PATH", "/data/raw/train.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot-topic-modeling")

DEFAULT_K = int(os.getenv("K", "6"))
DEFAULT_SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20000"))
DEFAULT_RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "TrustpilotTopicModel")


def clean_for_sbert(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="Nombre de clusters KMeans")
    parser.add_argument("--sample-size", type=int, default=None, help="Taille de l'échantillon")
    parser.add_argument("--random-state", type=int, default=None, help="Random state")
    return parser.parse_args()


def main():
    args = parse_args()

    k = args.k if args.k is not None else DEFAULT_K
    sample_size = args.sample_size if args.sample_size is not None else DEFAULT_SAMPLE_SIZE
    random_state = args.random_state if args.random_state is not None else DEFAULT_RANDOM_STATE

    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH, header=None, names=["label", "title", "text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    df["text_sbert"] = df["text"].apply(clean_for_sbert)

    if len(df) == 0:
        raise ValueError("Dataset vide après chargement")

    df_sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=random_state
    ).reset_index(drop=True)

    with mlflow.start_run(run_name=f"sbert_kmeans_k{k}"):
        start_time = time.time()

        log_params({
            "k": k,
            "sample_size": len(df_sample),
            "random_state": random_state,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "algo": "KMeans",
            "sklearn_version": sklearn.__version__
        })

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = embedder.encode(
            df_sample["text_sbert"].tolist(),
            show_progress_bar=False
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto"
        )
        _ = kmeans.fit_predict(embeddings)

        sil = silhouette_score(embeddings, kmeans.labels_)
        training_time = time.time() - start_time

        log_metric("silhouette_score", float(sil))
        log_metric("training_time_seconds", float(training_time))

        joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_topics.pkl"))
        joblib.dump(
            {i: f"Theme_{i}" for i in range(k)},
            os.path.join(MODEL_DIR, "cluster_labels.pkl")
        )

        mlflow.sklearn.log_model(
            sk_model=kmeans,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        log_artifacts(MODEL_DIR, artifact_path="exported_models")

    print({
        "status": "success",
        "silhouette": float(sil),
        "k": int(k),
        "sample_size": int(len(df_sample))
    })


if __name__ == "__main__":
    main()
