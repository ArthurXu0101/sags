#!/bin/bash

# Run Preprocess.py for image preprocessing
CUDA_VISIBLE_DEVICES=2 python Preprocess.py \
    --input_folder dataset \
    --output_folder /TEED/data
