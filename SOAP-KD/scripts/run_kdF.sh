#!/bin/bash

ROOT_PATH="Your_full_path/SOAP-KD/SOAP-KD_v2"
DATA_PATH="Your_full_path/FGSC-23_aligned_224x224.h5"
FAKE_DATA_PATH="${ROOT_PATH}/output/fake_data/FGSC23_fake_images_SAGAN_cDR-RS_precnn_resnet50_epochs_200_DR_CNN5_epochs_100_lambda_0.050_filter_mobilenet_v2_perc_0.70_adjust_True_Nlabel_0_NFakePerLabel_200_seed_2023.h5"

BETA=0.0 #set 1 if enable RKD

net_t_name="mobilenet_v2"
net_t_path="${ROOT_PATH}/output/CNN/vanilla/ckpt_${net_t_name}_epoch_200_pretrain_True_last.pth"

ALPHA=100.0
net_s_name="wrn_16_1"
net_s_path="${ROOT_PATH}/output/CNN/kd_vanilla/ckpt_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}_epoch_200_last.pth"
python main_kd.py --root_path $ROOT_PATH --data_path $DATA_PATH \
    --net_t_name $net_t_name --net_t_path $net_t_path --net_s_name $net_s_name \
    --alpha $ALPHA --beta $BETA \
    --use_fake_data --fake_data_path $FAKE_DATA_PATH \
    --finetune_net_s --init_s_path $net_s_path \
    2>&1 | tee output_kdF_SAGAN_S_${net_s_name}_T_${net_t_name}_a_${ALPHA}_b_${BETA}.txt