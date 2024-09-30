#!/bin/bash

# Run Preprocess.py to resize and convert images
CUDA_VISIBLE_DEVICES=2 python Preprocess.py \
    --input_folder /dataset \
    --output_folder /output/preprocessed_images \
    --max_resolution 1024

# Check if Preprocess.py was successful
if [ $? -ne 0 ]; then
    echo "Error: Preprocess.py failed."
    exit 1
fi

# Run Vectorize.py to process images and generate SVGs
CUDA_VISIBLE_DEVICES=2 python Vectorize.py \
    --dataset_path /output/preprocessed_images \
    --output_dir /output/vectorized_svgs \
    --n_clusters 12 \
    --epsilon 2.0 \
    --max_depth 5

# Check if Vectorize.py was successful
if [ $? -ne 0 ]; then
    echo "Error: Vectorize.py failed."
    exit 1
fi

echo "Both Preprocess.py and Vectorize.py have successfully run."
