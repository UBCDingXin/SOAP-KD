@echo off

set CUDA_VISIBLE_DEVICES=0

set ROOT_PATH="D:/local_wd/SOAP-KD_extra/NFD"
set DATA_PATH="C:/Users/DX/BaiduSyncdisk/Baidu_WD/datasets/remote_sensing/FGSC/FGSC-23_aligned/FGSC-23_aligned_224x224.h5"

set net_t_name="mobilenet_v2"
set net_t_path="./output/ckpt_%net_t_name%_epoch_200_pretrain_True_last.pth"

set net_s_name="resnet8"

python main_kd.py --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --net_t_name %net_t_name% --net_t_path %net_t_path% --net_s_name %net_s_name% ^ %*