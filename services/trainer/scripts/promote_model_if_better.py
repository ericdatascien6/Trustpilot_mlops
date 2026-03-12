import os
import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "TrustpilotTopicModel"
)

client = MlflowClient()

print("=== PROMOTION CHECK ===")


# récupérer toutes les versions existantes
latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)

if not latest_versions:
    raise RuntimeError("No model version found in registry")


# prendre la dernière version créée
candidate = max(latest_versions, key=lambda v: int(v.version))

candidate_version = candidate.version
candidate_run_id = candidate.run_id

candidate_metric = client.get_run(candidate_run_id).data.metrics.get(
    "silhouette_score"
)

print("candidate_version :", candidate_version)
print("candidate_run_id :", candidate_run_id)
print("candidate_metric :", candidate_metric)


# essayer de récupérer la version Production
try:

    prod = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        "Production"
    )

    prod_version = prod.version
    prod_run_id = prod.run_id

    prod_metric = client.get_run(prod_run_id).data.metrics.get(
        "silhouette_score"
    )

    print("current_production_version :", prod_version)
    print("current_production_metric :", prod_metric)


except Exception:

    print("No Production model yet → promoting first model")

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "Production",
        candidate_version
    )

    print(
        "PROMOTED version",
        candidate_version,
        "to alias Production"
    )

    exit()


# comparaison des performances
if candidate_metric >= prod_metric:

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "Production",
        candidate_version
    )

    print(
        "PROMOTED version",
        candidate_version,
        "to alias Production"
    )

else:

    print("No promotion needed")
