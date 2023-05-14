#!/bin/bash

ROOT_PATH="Your_full_path/SOAP-KD/DKD"
DATA_PATH="Your_full_path/FGSC-23_aligned_224x224.h5"


cnn_name="mobilenet_v2"
python main_vanilla.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --cnn_name $cnn_name --pretrained \
    2>&1 | tee output_baseline_${cnn_name}.txt