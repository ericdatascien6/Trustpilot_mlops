#!/bin/bash

echo "Starting MLflow server..."

docker-compose up -d mlflow

echo "MLflow is running at:"
echo "http://localhost:5000"

docker ps | grep mlflow
