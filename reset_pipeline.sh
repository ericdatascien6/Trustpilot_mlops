#!/bin/bash

echo "Resetting pipeline state..."

echo 0 > data/metadata/stream_offset.txt

echo "label,title,text" > data/sas/trustpilot_new_reviews.csv

head -1000 data/dataset_source/amazon_reviews.csv > data/raw/train.csv

echo "Reset done."
