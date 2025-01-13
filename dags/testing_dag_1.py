from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Initialize the DAG
with DAG(
    "basic_task_logging_dag",
    default_args=default_args,
    description="A simple DAG with logging to test Airflow setup",
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    def task_log_1(**kwargs):
        """Task 1 logs."""
        print("Task 1: Logging some info... Airflow setup works!")

    def task_log_2(**kwargs):
        """Task 2 logs."""
        print("Task 2: Another log... Everything is fine!")

    def task_log_3(**kwargs):
        """Task 3 logs."""
        print("Task 3: Final task log... DAG structure is valid!")

    # Define tasks
    task_1 = PythonOperator(
        task_id="task_log_1",
        python_callable=task_log_1,
    )

    task_2 = PythonOperator(
        task_id="task_log_2",
        python_callable=task_log_2,
    )

    task_3 = PythonOperator(
        task_id="task_log_3",
        python_callable=task_log_3,
    )

    # Set task dependencies
    task_1 >> task_2 >> task_3
