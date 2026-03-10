#!/bin/bash

echo "Stopping containers..."
docker compose down

echo "Cleaning Airflow logs..."
sudo rm -rf airflow/logs/*
sudo mkdir -p airflow/logs
sudo chown -R 50000:50000 airflow/logs

echo "Cleaning MLflow runs..."
sudo rm -rf mlruns/*
mkdir -p mlruns

echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup finished."
