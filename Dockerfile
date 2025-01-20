FROM apache/airflow:2.7.3-python3.10

USER root

RUN mkdir -p /mlflow
RUN chmod -R 777 /mlflow

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

