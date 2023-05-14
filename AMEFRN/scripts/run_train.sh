#!/bin/bash

ROOT_PATH="Your_full_path/SOAP-KD/AMEFRN"
DATA_PATH="Your_full_path/FGSC-23_aligned_resplit_224x224.h5"

python main.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --cnn_name vgg16 \
    2>&1 | tee output_baseline_vgg16.txt