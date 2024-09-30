#!/bin/bash

# Set CUDA device and run the Python script
CUDA_VISIBLE_DEVICES=2 python Vectorize.py \
    --dataset_path /Vectorize/TEED/data/ \
    --edge_dir /TEED/result/BIPED2CLASSIC/fused \
    --output_dir /output/output_svgs/ \
    --n_clusters 32 \
    --epsilon 1.5 \
    --max_depth 7 \
    --threshold 0.4
