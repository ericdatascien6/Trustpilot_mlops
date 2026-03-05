#!/bin/bash

echo "Starting API service..."

docker-compose up -d api

echo ""
echo "API should be available at:"
echo "http://localhost:8000"
echo ""

echo "API container status:"
docker ps | grep trustpilot_api
