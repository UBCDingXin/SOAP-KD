#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT_PATH="/home/xin/WD/SOAP-KD_extra/NFD"
DATA_PATH="/home/xin/WD/datasets/remote_sensing/FGSC/FGSC-23_aligned/FGSC-23_aligned_224x224.h5"

net_t_name="mobilenet_v2"
net_t_path="${ROOT_PATH}/output/ckpt_${net_t_name}_epoch_200_pretrain_True_last.pth"

net_s_name="resnet8"

python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}.txt