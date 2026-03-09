import os
import joblib
import numpy as np
import re
import mlflow
import mlflow.sklearn
from sentence_transformers import SentenceTransformer
from pathlib import Path
from mlflow.tracking import MlflowClient

# connexion au serveur MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

MODELS_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_NAME = "TrustpilotTopicModel"
MODEL_ALIAS = "Production"

print("=== MLflow Registry Evaluation ===")
print("model :", MODEL_NAME)
print("alias :", MODEL_ALIAS)

client = MlflowClient()

# récupération du modèle en Production
try:
    prod_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    print("--- Production ---")
    print("version :", prod_version.version)
    print("run_id  :", prod_version.run_id)

    run = client.get_run(prod_version.run_id)

    metric = run.data.metrics.get("silhouette_score")
    print("metric  :", metric)

    # chargement du modèle Production
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    print("model loaded successfully")

except Exception as e:

    print("--- Production ---")
    print("No Production model found")
    print("error :", str(e))
    model = None

# chargement des labels des clusters
try:
    cluster_labels = joblib.load(MODELS_DIR / "cluster_labels.pkl")
    print("cluster_labels loaded")
except Exception as e:
    print("cluster_labels not found :", str(e))

# chargement du modèle SBERT
try:
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("SBERT model loaded")
except Exception as e:
    print("SBERT loading error :", str(e))

print("--- Decision (theoretical) ---")
print("should_promote: False")
