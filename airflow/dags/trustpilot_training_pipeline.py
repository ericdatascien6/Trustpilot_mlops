from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops",
    "retries": 1
}

with DAG(
    dag_id="trustpilot_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command="docker run --rm --network trustpilot_mlops_default -v /home/ubuntu/Trustpilot_mlops/data:/data trustpilot_mlops_trainer python train_job.py --k 6"
    )

    evaluate_model = BashOperator(
        task_id="evaluate_registry",
        bash_command="docker run --rm --network trustpilot_mlops_default trustpilot_mlops_trainer python /app/scripts/evaluate_registry.py"
    )

    train_model >> evaluate_model
