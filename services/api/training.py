import os
import re
import joblib
import pandas as pd

import mlflow
from mlflow import log_params, log_metric, log_artifacts

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==============================
# Variables d’environnement
# ==============================

DATA_PATH = os.getenv("DATA_PATH", "/data/raw/train.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot-topic-modeling")

K = int(os.getenv("K", "6"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20000"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


# ==============================
# Nettoyage texte
# ==============================

def clean_for_sbert(text: str) -> str:
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
# Fonction appelée par l'API
# ==============================

def train_model():
    """
    Entraîne un modèle SBERT + KMeans
    Sauvegarde les artefacts dans MODEL_DIR
    Log les métriques dans MLflow
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Chargement des données
    df = pd.read_csv(DATA_PATH, header=None, names=["label", "title", "text"])
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    df["text_sbert"] = df["text"].apply(clean_for_sbert)

    # Sampling pour accélérer l’entraînement
    df_sample = df.sample(
        min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    with mlflow.start_run(run_name="sbert_kmeans_training"):

        # Log des paramètres
        log_params({
            "k": K,
            "sample_size": len(df_sample),
            "random_state": RANDOM_STATE,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "algo": "KMeans"
        })

        # Encodage SBERT
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedder.encode(
            df_sample["text_sbert"].tolist(),
            show_progress_bar=True
        )

        # Clustering
        kmeans = KMeans(
            n_clusters=K,
            random_state=RANDOM_STATE,
            n_init="auto"
        )
        labels = kmeans.fit_predict(embeddings)

        # Évaluation
        sil = silhouette_score(embeddings, labels)
        log_metric("silhouette", float(sil))

        # Export modèles pour l’inférence
        joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_topics.pkl"))
        joblib.dump(
            {i: f"Theme_{i}" for i in range(K)},
            os.path.join(MODEL_DIR, "cluster_labels.pkl")
        )

        # Log artefacts MLflow
        log_artifacts(MODEL_DIR, artifact_path="exported_models")

    # Retour API
    return {
        "status": "success",
        "silhouette": float(sil),
        "k": K,
        "sample_size": len(df_sample)
    }
