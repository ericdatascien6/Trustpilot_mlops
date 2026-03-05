#!/bin/bash

echo "Running API tests..."

docker-compose run --rm api-tests

echo "Tests finished."
