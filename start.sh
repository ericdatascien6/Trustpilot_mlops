#!/bin/bash

echo "================================="
echo "Starting Trustpilot MLOps stack"
echo "================================="

echo ""
echo "1. Building images..."
docker-compose build

echo ""
echo "2. Starting containers..."
docker-compose up -d

echo ""
echo "3. Containers status:"
docker-compose ps

echo ""
echo "================================="
echo "Stack started successfully"
echo ""
echo "MLflow UI   : http://localhost:5000"
echo "Airflow UI  : http://localhost:8081"
echo "Login       : admin / admin"
echo "================================="
