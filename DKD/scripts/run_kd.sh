#!/bin/bash

ROOT_PATH="Your_full_path/SOAP-KD/DKD"
DATA_PATH="Your_full_path/FGSC-23_aligned_224x224.h5"


ALPHA=0.0
BETA=0.0

#############################################################
net_t_name="mobilenet_v2"
net_t_path="${ROOT_PATH}/output/CNN/vanilla/ckpt_${net_t_name}_epoch_200_pretrain_True_last.pth"

net_s_name="resnet8"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA --lr_base 1e-3 \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt

net_s_name="resnet14"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA --lr_base 1e-3 \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt

net_s_name="wrn_16_1"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA --lr_base 1e-3 \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt

net_s_name="shufflenet_v2_x0_5"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA --lr_base 1e-3 \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt

net_s_name="shufflenet_v2_x1_0"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA --lr_base 1e-3 \
    2>&1 | tee output_kd_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt