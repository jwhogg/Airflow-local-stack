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
import torch
import sklearn
import mlflow
from mlflow.tracking import MlflowClient
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger("airflow.task")

# MinIO configuration
MINIO_URL = "minio:9000"
MINIO_ACCESS_KEY = Variable.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = Variable.get("MINIO_SECRET_KEY")
BUCKET_NAME = "datasets"

FEATURE_DATE = "2025-01-20"
FEATURE_VERSION = "1"

MLFLOW_TRACKING_URI = "http://mlflow:7100"
MLFLOW_EXPERIMENT_NAME = "energy-price-prediction"
MODEL_REGISTRY_NAME = "energy_price_predictor"

# Model Configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 5

# MinIO client
client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def save_df_as_csv_to_minio(df, fileName, idx=False):
    csv_bytes = df.to_csv(index=idx).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    client.put_object(
        BUCKET_NAME, fileName, data=csv_buffer, length=len(csv_bytes), content_type="application/csv"
    )

def load_feature(filename: str, bucket='datasets'):
    response = client.get_object(bucket, filename)
    try:
        file_content = BytesIO(response.read())
        file_content.seek(0)
        tensor = torch.load(file_content)
    finally:
        response.close()
        response.release_conn()
    
    return tensor

class EnergyPriceLSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

def train_model(**context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()
    
    with mlflow.start_run() as run:
        # Load training data
        train_features = load_feature(f"train_features_v{FEATURE_VERSION}_{FEATURE_DATE}.pt", BUCKET_NAME)
        train_labels = load_feature(f"train_labels_v{FEATURE_VERSION}_{FEATURE_DATE}.pt", BUCKET_NAME)
        val_features = load_feature(f"val_features_v{FEATURE_VERSION}_{FEATURE_DATE}.pt", BUCKET_NAME)
        val_labels = load_feature(f"val_labels_v{FEATURE_VERSION}_{FEATURE_DATE}.pt", BUCKET_NAME)
        
        # Create data loaders
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataset = TensorDataset(val_features, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model and training parameters
        model = EnergyPriceLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Log parameters
        mlflow.log_params({
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        })
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = model(batch_features)
                    val_loss += criterion(outputs, batch_labels).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, step=epoch)
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Log model to MLflow Model Registry
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    registered_model_name=MODEL_REGISTRY_NAME
                )
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Transition model to Production if it's the best so far
        client = MlflowClient()
        models = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        current_model = models[0]  # Latest version
        
        # Compare with previous production model if exists
        prod_model = None
        for model in models:
            if model.current_stage == 'Production':
                prod_model = model
                break
        
        if not prod_model or best_val_loss < float(prod_model.tags.get('val_loss', 'inf')):
            # Transition this model to production
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=current_model.version,
                stage="Production"
            )
            # Archive previous production model if it exists
            if prod_model:
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY_NAME,
                    version=prod_model.version,
                    stage="Archived"
                )
        
        # Save run info for evaluation
        context['task_instance'].xcom_push(key='run_id', value=run.info.run_id)

def evaluate_model(**context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    with mlflow.start_run(run_id=context['task_instance'].xcom_pull(
            task_ids='train_model', key='run_id')):
            
        # Load the production model from the registry
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{MODEL_REGISTRY_NAME}/Production"
        )
        
        # Load validation data
        val_features = load_feature(f"val_features_v{FEATURE_VERSION}_{FEATURE_DATE}.pt", BUCKET_NAME)
        val_labels = load_feature(f"val_labels_v{FEATURE_VERSION}_{FEATURE_DATE}.pt", BUCKET_NAME)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(val_features)
        
        # Calculate metrics
        mse = mean_squared_error(val_labels, predictions)
        mae = mean_absolute_error(val_labels, predictions)
        rmse = np.sqrt(mse)
        
        # Log metrics
        mlflow.log_metrics({
            "test_mse": mse,
            "test_mae": mae,
            "test_rmse": rmse
        })

# Define the DAG
with DAG(
    dag_id="ML_EPF__Model_Pipeline",
    schedule_interval=None,
    description='ML pipeline for training a model on Electricity Forecasting dataset from kaggle',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        dag=dag
    )

    train_model_task >> evaluate_model_task
