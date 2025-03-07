How to make a local stack from scatch:
(If pulling the repo, check the docker files have the right variables for you, and skip to step 5)

1. make a new folder
2. `touch Dockerfile`
3. contents of dockerfile:
```Dockerfile
FROM apache/airflow:latest

RUN pip install pandas

USER airflow
```
4. touch docker-compose.yml:
```yml
services:
  postgres:
    image: postgres:latest
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5

  webserver:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__LOAD_EXAMPLES=false
      - AIRFLOW__WEBSERVER__SECRET_KEY=your-secret-key-here
      - AIRFLOW__WEBSERVER__BASE_URL=http://localhost:8080
      - AIRFLOW__WEBSERVER__WEB_SERVER_HOST=0.0.0.0
      - AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8080
    # command: webserver
    command: >
      bash -c "
        airflow db init &&
        airflow users create --username admin --password admin --firstname joel --lastname hogg --role Admin --email joelthehogg@gmail.com &&
        airflow webserver
      "
    volumes:
      - ./dags:/opt/airflow/dags
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__LOAD_EXAMPLES=false
    command: scheduler
    volumes:
      - ./dags:/opt/airflow/dags
```
5. `docker compose build`
### Running it
`docker compose up -d`
go to https://localhost:8080
username: admin, password: admin

Make any .py files under the 'Dags' directory

### Important!
- if you've added a python package to the requirements.txt, you need to run `docker-compose build --no-cache`
