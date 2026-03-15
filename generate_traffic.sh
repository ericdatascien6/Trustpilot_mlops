#!/bin/bash

echo "Generating API traffic..."

texts=(
"This book is excellent and very fast delivered"
"The movie was amazing and very emotional"
"This product is terrible and broke after one day"
"Fantastic music album with great sound quality"
"The documentary was very interesting"
"Delivery was late but the product is good"
"Great customer service and fast response"
"The quality of the item is disappointing"
"Amazing story and great actors"
"This is one of the best purchases I made"
)

while true
do
  random_index=$((RANDOM % ${#texts[@]}))
  text="${texts[$random_index]}"

  curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: secret123" \
  -d "{\"text\":\"$text\"}" > /dev/null

  echo "Prediction sent: $text"

  sleep 1
done
