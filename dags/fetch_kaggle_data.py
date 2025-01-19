from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
import logging
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO

logger = logging.getLogger("airflow.task")

# Function to authenticate Kaggle, download the dataset, and load it into a DataFrame
def download_and_load_data():
    # Set the environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = Variable.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = Variable.get("KAGGLE_KEY")
    
    import kaggle

    # Authenticate using Kaggle API
    kaggle.api.authenticate()

    logger.info("Authenticated with Kaggle correctly")

    logger.info("Downloading files...")

    # Download the dataset from Kaggle
    kaggle.api.dataset_download_files("nicholasjhana/energy-consumption-generation-prices-and-weather", path='.', unzip=True)

    energy_df = pd.read_csv('energy_dataset.csv')

    # Load the weather features dataset
    weather_df = pd.read_csv('weather_features.csv')

    logger.info("Files downloaded successfully!")

    print(energy_df.head())
    print(weather_df.head())

    from minio import Minio

    client = Minio("minio:9000",
        access_key="5MpW7tf4J1U4KKvzLm7X",
        secret_key="bOpLlAAeSqJgfZBvLZFNRZA9o1eP5Mc7l7ea8R4V",
        secure=False,
    )

    csv_bytes = energy_df.to_csv().encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    logger.info("Attempting to store datasets to MinIO...")

    client.put_object(
        "datasets", "energy.csv", data=csv_buffer, length=len(csv_bytes), content_type="application/csv"
    )

    csv_bytes = weather_df.to_csv().encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    client.put_object(
        "datasets", "weather.csv", data=csv_buffer, length=len(csv_bytes), content_type="application/csv"
    )

# Define the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'kaggle_data_download',
    default_args=default_args,
    description='Download and load Kaggle data into pandas',
    schedule_interval=None,  # Set your schedule interval here (None means manual execution)
    start_date=datetime(2025, 1, 12),
    catchup=False,
) as dag:
    
    # Define the task that runs the Python function
    download_task = PythonOperator(
        task_id='download_and_load_data',
        python_callable=download_and_load_data,
    )

# The task will be executed when triggered, no dependencies are defined, as it's a simple single task.
