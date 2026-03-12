import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot-topic-modeling")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "TrustpilotTopicModel")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.silhouette_score DESC"]
)

best_run = None

for run in runs:

    run_id = run.info.run_id

    try:
        artifacts = client.list_artifacts(run_id, "model")
    except Exception:
        continue

    if not artifacts:
        print(f"Skipping run {run_id} (no model artifact)")
        continue

    ### AJOUT : vérification réelle que le modèle existe
    model_path = f"/mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"

    if not os.path.exists(model_path):
        print(f"Skipping run {run_id} (model path missing)")
        continue

    ### si on arrive ici → run valide
    best_run = run
import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot-topic-modeling")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "TrustpilotTopicModel")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found")

# On récupère uniquement les runs terminés
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.silhouette_score DESC"]
)

best_run = None

for run in runs:

    run_id = run.info.run_id

    try:
        artifacts = client.list_artifacts(run_id)
    except Exception:
        continue

    # vérifier que le dossier "model" existe dans les artifacts
    model_exists = any(a.path == "model" for a in artifacts)

    if not model_exists:
        print(f"Skipping run {run_id} (no model artifact)")
        continue

    # Vérification locale (sécurité supplémentaire)
    model_path = f"/mlruns/{experiment.experiment_id}/{run_id}/artifacts/model"

    if not os.path.exists(model_path):
        print(f"Skipping run {run_id} (model path missing)")
        continue

    best_run = run
    break


if best_run is None:
    raise RuntimeError("No valid run with model artifact found")


run_id = best_run.info.run_id
k = best_run.data.params.get("k")
silhouette = best_run.data.metrics.get("silhouette_score")

model_uri = f"runs:/{run_id}/model"

print("=== BEST MODEL ===")
print("run_id :", run_id)
print("k :", k)
print("silhouette :", silhouette)
print("model_uri :", model_uri)

print("registering model to registry...")

result = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)

print("registered version :", result.version)
