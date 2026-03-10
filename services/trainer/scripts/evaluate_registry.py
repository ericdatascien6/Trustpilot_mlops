import os
import mlflow
from mlflow.tracking import MlflowClient

dag_run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID")

mlflow.set_tracking_uri("http://mlflow:5000")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trustpilot-topic-modeling")

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.airflow_run = '{dag_run_id}'",
    order_by=["metrics.silhouette_score DESC"],
    max_results=50
)

if len(runs) == 0:
    raise ValueError("No runs found")

best_run = runs[0]

best_run_id = best_run.info.run_id
best_silhouette = best_run.data.metrics.get("silhouette_score")
best_k = best_run.data.params.get("k")

print("=== BEST MODEL ===")
print("run_id :", best_run_id)
print("k :", best_k)
print("silhouette :", best_silhouette)

model_uri = f"runs:/{best_run_id}/model"
print("model_uri :", model_uri)


MODEL_NAME = "TrustpilotTopicModel"

print("registering model to registry...")

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("registered version :", result.version)
