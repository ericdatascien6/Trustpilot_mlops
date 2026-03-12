from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


with DAG(
    dag_id="trustpilot_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    k_values = [3, 4, 5, 6, 7, 8]

    train_tasks = []

    for k in k_values:

        task = BashOperator(
            task_id=f"train_k{k}",
            bash_command=f"""
            docker run --rm \
            --network trustpilot_mlops_default \
            -v /home/ubuntu/Trustpilot_mlops/data:/data \
            -v /home/ubuntu/Trustpilot_mlops/mlruns:/mlruns \
            -v /home/ubuntu/Trustpilot_mlops/models:/models \
            trustpilot_mlops_trainer \
            python train_job.py --k {k}
            """
        )

        train_tasks.append(task)


    evaluate_model = BashOperator(
        task_id="evaluate_registry",
        bash_command="""
        docker run --rm \
        --network trustpilot_mlops_default \
        -v /home/ubuntu/Trustpilot_mlops/mlruns:/mlruns \
        trustpilot_mlops_trainer \
        python /app/scripts/evaluate_registry.py
        """
    )


    promote_model = BashOperator(
        task_id="promote_model_if_better",
        bash_command="""
        docker run --rm \
        --network trustpilot_mlops_default \
        -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
        -v /home/ubuntu/Trustpilot_mlops/mlruns:/mlruns \
        trustpilot_mlops_trainer \
        python /app/scripts/promote_model_if_better.py
        """
    )


    simulate_stream = BashOperator(
        task_id="simulate_review_stream",
        bash_command="""
        cd /opt/airflow && \
        python3 services/trainer/scripts/simulate_review_stream.py
        """
    )


    update_dataset = BashOperator(
        task_id="update_training_dataset",
        bash_command="""
        cd /opt/airflow && \
        python3 services/trainer/scripts/update_training_dataset.py
        """
    )


    simulate_stream >> update_dataset >> train_tasks >> evaluate_model >> promote_model
