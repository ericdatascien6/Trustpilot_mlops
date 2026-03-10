#!/bin/bash

echo "================================="
echo "Testing Trustpilot Inference API"
echo "================================="

curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-H "x-api-key: secret123" \
-d '{
"text": "This book is excellent and very fast delivered"
}'

echo ""
echo ""
echo "Inference request sent."
