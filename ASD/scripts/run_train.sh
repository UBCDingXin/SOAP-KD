#!/bin/bash

ROOT_PATH="Your_full_path/SOAP-KD/ASD"
DATA_PATH="Your_full_path/FGSC-23_aligned_224x224.h5"


cnn_name="vgg16"
n_cls=60
python main.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --cnn_name $cnn_name --num_classes $n_cls \
    2>&1 | tee output_baseline_${cnn_name}_${n_cls}.txt