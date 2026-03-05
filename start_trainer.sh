#!/bin/bash

echo "Starting trainer job..."

docker-compose run --rm trainer python train_job.py

echo "Training finished."
