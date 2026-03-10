import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "TrustpilotTopicModel")
METRIC_NAME = "silhouette_score"

client = MlflowClient()

latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if len(latest_versions) == 0:
    raise ValueError("No registered model versions found")

candidate = max(latest_versions, key=lambda mv: int(mv.version))
candidate_run = client.get_run(candidate.run_id)
candidate_metric = candidate_run.data.metrics.get(METRIC_NAME)

if candidate_metric is None:
    raise ValueError(f"Metric '{METRIC_NAME}' not found for candidate version")

try:
    prod = client.get_model_version_by_alias(MODEL_NAME, "Production")
    prod_run = client.get_run(prod.run_id)
    prod_metric = prod_run.data.metrics.get(METRIC_NAME)
except Exception:
    prod = None
    prod_metric = None

print("=== PROMOTION CHECK ===")
print("candidate_version :", candidate.version)
print("candidate_run_id :", candidate.run_id)
print("candidate_metric :", candidate_metric)

if prod is not None:
    print("current_production_version :", prod.version)
    print("current_production_metric :", prod_metric)
else:
    print("current_production_version : None")
    print("current_production_metric : None")

if prod is None or prod_metric is None or candidate_metric > prod_metric:
    client.set_registered_model_alias(MODEL_NAME, "Production", candidate.version)
    print(f"PROMOTED version {candidate.version} to alias Production")
else:
    print("NO PROMOTION")
