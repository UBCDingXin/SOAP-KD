#!/bin/bash

ROOT_PATH="/scratch/dingx92/SOAP-KD/SOAP-KD_v2"
DATA_PATH="/project/6000538/dingx92/datasets/FGSC-23_aligned/FGSC-23_aligned_224x224.h5"

BETA=0.0

net_t_name="mobilenet_v2"
net_t_path="${ROOT_PATH}/output/CNN/vanilla/ckpt_${net_t_name}_epoch_200_pretrain_True_last.pth"

ALPHA=100.0

net_s_name="wrn_16_1"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt