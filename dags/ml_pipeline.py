from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
import logging
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger("airflow.task")

# MinIO configuration
MINIO_URL = "minio:9000"
MINIO_ACCESS_KEY = Variable.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = Variable.get("MINIO_SECRET_KEY")
BUCKET_NAME = "datasets"

# File configuration
KAGGLE_DATASET = "nicholasjhana/energy-consumption-generation-prices-and-weather"
DATASET_FILES = ["energy_dataset.csv", "weather_features.csv"]
os.environ["KAGGLE_USERNAME"] = Variable.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = Variable.get("KAGGLE_KEY")
import kaggle #need to import kaggle after env variables set
kaggle.api.authenticate()

# MinIO client
client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# Ensure the bucket exists
if not client.bucket_exists(BUCKET_NAME):
    client.make_bucket(BUCKET_NAME)


def download_from_kaggle():
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=False)
    logger.info(os.listdir(os.getcwd()))
    with zipfile.ZipFile('energy-consumption-generation-prices-and-weather.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    logger.info(os.listdir(os.getcwd()))



def save_df_as_csv_to_minio(df, fileName):
    csv_bytes = df.to_csv().encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    client.put_object(
        BUCKET_NAME, fileName, data=csv_buffer, length=len(csv_bytes), content_type="application/csv"
    )

def download_dataset():
    download_from_kaggle()

    logger.info("Files downloaded successfully!")

    energy_df = pd.read_csv('energy_dataset.csv')
    weather_df = pd.read_csv('weather_features.csv')
    
    for fileName in DATASET_FILES:
        df = pd.read_csv(fileName)
        if not file_exists(client, BUCKET_NAME, fileName):
            save_df_as_csv_to_minio(df, fileName)
            logger.info(f"File {fileName} saved successfully!")
        else:
            logger.info(f"File {fileName} already exists!")

def fetch_from_minio(filename, bucket_name):

    response = client.get_object(bucket_name, filename)
    
    try:
        file_content = BytesIO(response.read())
        df = pd.read_csv(file_content)
    finally:
        response.close()
        response.release_conn()
    return df

def preprocess_data(**kwargs):
    # Preprocess the file
    df_list = [fetch_from_minio(filename, BUCKET_NAME) for filename in DATASET_FILES]

    #put preprocessing in here...
    for df in df_list:
        print(df.head())

def file_exists(client: Minio, bucket_name: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise



# Define the DAG
with DAG(
    dag_id="ML_EPF_Pipeline",
    schedule_interval=None,
    description='ML pipeline for training a model on Electricity Forecasting dataset from kaggle',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Download from Kaggle
    download_task = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
    )

    # Task 2: Preprocess Data
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    # Task dependencies
    download_task >> preprocess_task
