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
import numpy as np

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



def save_df_as_csv_to_minio(df, fileName, idx=False):
    csv_bytes = df.to_csv(index=idx).encode('utf-8')
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

def fetch_from_minio(filename, bucket_name, idx=False):

    response = client.get_object(bucket_name, filename)
    
    try:
        file_content = BytesIO(response.read())
        df = pd.read_csv(file_content, index_col=idx)
    finally:
        response.close()
        response.release_conn()
    return df

def preprocess_data(**kwargs):
    # Preprocess the file
    df_list = [fetch_from_minio(filename, BUCKET_NAME) for filename in DATASET_FILES]

    for i, df in enumerate(df_list): #this for is only 2 items

        # Drop the columns that only contain Nones and/or NaNs and all 0s
        for col in df:
            if (
                df[col].isnull().all() or 
                (df[col].dtype != "datetime64[ns, UTC]" and df[col].sum() == 0)
            ):
                df = df.drop(columns=[col], axis=1)
        
        # Set the time to datetime type and set as index
        if "dt_iso" in df.columns:
            df['time'] = pd.to_datetime(df['dt_iso'], utc=True, infer_datetime_format=True)
            df = df.drop(['dt_iso'], axis=1) #not getting dropped here
            df = df.set_index('time')
        elif "time" in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True, infer_datetime_format=True)
            df = df.set_index('time')

        #check for duplicates

        duplicate_row_count = df.duplicated(keep='first').sum()
        if duplicate_row_count > 0:
            df = df.reset_index().drop_duplicates(subset=['time', 'city_name'],
                                                    keep='first').set_index('time')

        #convert any ints to floats
        cols = df.select_dtypes(include=[np.int64]).columns
        for col in cols:
            df[col] = df[col].values.astype(np.float64)
        
        #drop any textual columns
        df = df.drop(columns=[col for col in df.columns if df[col].dtypes == 'object' and col != "city_name"])

        #remove extreme pressure outliers for weather df
        if "pressure" in df.columns:
            df.loc[df.pressure > 1051, 'pressure'] = np.nan
            df.loc[df.pressure < 931, 'pressure'] = np.nan
        
        if "wind_speed" in df.columns:
            df.loc[df.wind_speed > 50, 'wind_speed'] = np.nan

        #interpolate any null values
        null_value_count = df.isnull().values.sum()
        if null_value_count > 0:
            df.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

        df_list[i] = df

    # seperate weather dfs
    weather_df = df_list[1]
    energy_df = df_list[0]
    if "rain_3h" in weather_df.columns:
        weather_df = weather_df.drop(['rain_3h'], axis=1) #drop unreliable feature
    weather_df_1, weather_df_2, weather_df_3, weather_df_4, weather_df_5 = [x for _, x in weather_df.groupby('city_name')]
    weather_dfs = [weather_df_1, weather_df_2, weather_df_3, weather_df_4, weather_df_5]

    #merge into one df
    df_final = energy_df
    df_final = df_final.round(3)

    for df in weather_dfs:
        city = df['city_name'].unique()
        city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
        df = df.add_suffix('_{}'.format(city_str))
        df_final = df_final.merge(df, on=['time'], how='outer')
        df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)

    #save preprocessed df to minio
    save_df_as_csv_to_minio(df_final, "merged_preprocessed.csv", idx=True)

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
