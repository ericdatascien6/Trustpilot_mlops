import argparse
from mlflow.tracking import MlflowClient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--registered-model-name", default="TrustpilotTopicModel")
    p.add_argument("--alias", default="Production")
    p.add_argument("--metric-name", default="silhouette_score")
    p.add_argument("--top-n", type=int, default=10)
    return p.parse_args()


def safe_metric(client: MlflowClient, run_id: str, metric_name: str):
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_name)


def main():
    args = parse_args()
    client = MlflowClient()

    # 1) Production
    try:
        prod_mv = client.get_model_version_by_alias(args.registered_model_name, args.alias)
        prod_version = int(prod_mv.version)
        prod_metric = safe_metric(client, prod_mv.run_id, args.metric_name)
    except Exception:
        prod_mv = None
        prod_version = None
        prod_metric = None

    # 2) Dernières versions (pour trouver la meilleure)
    versions = client.search_model_versions(f"name='{args.registered_model_name}'")
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)[: args.top_n]

    best_version = None
    best_metric = None

    for v in versions_sorted:
        m = safe_metric(client, v.run_id, args.metric_name)
        if m is None:
            continue
        if (best_metric is None) or (m > best_metric):
            best_metric = m
            best_version = int(v.version)

    # 3) Décision "théorique"
    should_promote = False
    if best_metric is not None:
        if prod_metric is None:
            should_promote = True
        else:
            should_promote = best_metric > prod_metric

    print("=== MLflow Registry Evaluation ===")
    print(f"model                 : {args.registered_model_name}")
    print(f"metric                : {args.metric_name}")
    print(f"checked_last_versions : {len(versions_sorted)}")

    print("--- Production ---")
    print(f"alias       : {args.alias}")
    print(f"version     : {prod_version}")
    print(f"metric      : {prod_metric}")

    print("--- Best found ---")
    print(f"best_version: {best_version}")
    print(f"best_metric : {best_metric}")

    print("--- Decision (theoretical) ---")
    print(f"should_promote: {should_promote}")


if __name__ == "__main__":
    main()
