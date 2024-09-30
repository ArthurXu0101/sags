#!/bin/bash

# Set CUDA device and run the Python script
CUDA_VISIBLE_DEVICES=2 python edge_processing.py \
    --image_dir "/dataset" \
    --svg_dir "/output/output_svgs/" \
    --output_dir "/output/output_visualizations" \
    --edge_dir "/output/edges/original_edges" \
    --svgEdge_folder "/output/edges/svg_edges" \
    --tolerance 5
