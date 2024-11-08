FROM apache/airflow:2.9.2-python3.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the /app/final_models directory and set permissions
RUN mkdir -p /app/final_models && chmod -R 777 /app/final_models

# Switch back to the airflow user
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
