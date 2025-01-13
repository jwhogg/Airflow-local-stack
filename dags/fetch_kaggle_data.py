from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta

# Function to authenticate Kaggle, download the dataset, and load it into a DataFrame
def download_and_load_data():
    # Set the environment variables for Kaggle authentication
    os.environ["KAGGLE_USERNAME"] = Variable.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = Variable.get("KAGGLE_KEY")
    
    import kaggle

    # Authenticate using Kaggle API
    kaggle.api.authenticate()

    # Download the dataset from Kaggle
    kaggle.api.dataset_download_files("nicholasjhana/energy-consumption-generation-prices-and-weather", path='.', unzip=True)

    # Extract the downloaded zip file
    with zipfile.ZipFile('energy-consumption-generation-prices-and-weather.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    # Assuming the dataset includes a CSV file (adjust filename if necessary)
    csv_file = 'energy-consumption-generation-prices-and-weather.csv'  # Adjust this to the actual file name
    
    # Load the data into pandas
    df = pd.read_csv(csv_file)

    # Print the head of the dataframe (or save it for further use)
    print(df.head())

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
