#!/bin/bash

echo "Stopping stack..."
docker-compose down

echo "Cleaning MLflow..."
rm -rf mlruns/*

echo "Restarting stack..."
docker-compose up -d

echo "MLflow reset complete"
