#!/bin/bash

ROOT_PATH="Your_full_path/SOAP-KD/SOAP-KD_v2"
DATA_PATH="Your_full_path/FGSC-23_aligned_224x224.h5"
TEACHER_MODEL_PATH=${ROOT_PATH}/output/CNN/vanilla

KAPPA=-2

python gene_fake_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" \
    --gan_gene_ch 32 --gan_disc_ch 32 \
    --gan_embed_x2y_epoch 10 \
    --gan_niters 40000 --gan_resume_niters 0 --gan_save_niters_freq 5000 \
    --gan_d_niters 2 --gan_lr_g 1e-4 --gan_lr_d 1e-4 \
    --gan_batch_size_disc 100 --gan_batch_size_gene 100 --gan_batch_size_vis 10 \
    --num_grad_acc_d 1 --num_grad_acc_g 1 \
    --gan_threshold_type "soft" --gan_kappa $KAPPA \
    --gan_DiffAugment \
    --samp_batch_size 200 --samp_burnin_size 200 \
    --samp_num_fake_labels 0 --samp_nfake_per_label 200 \
    --subsampling \
    --dre_precnn_name resnet50 --dre_precnn_epochs 200 \
    --dre_epochs 100 --dre_lambda 0.05 \
    2>&1 | tee output_gene_SAGAN_subsampling.txt

net_t_name="mobilenet_v2"
net_t_path="${TEACHER_MODEL_PATH}/ckpt_${net_t_name}_epoch_200_pretrain_True_last.pth"
unfiltered_fake_dataset_path="${ROOT_PATH}/output/fake_data/FGSC23_fake_images_SAGAN_cDR-RS_precnn_resnet50_epochs_200_DR_CNN5_epochs_100_lambda_0.050_filter_None_adjust_False_Nlabel_0_NFakePerLabel_200_seed_2023.h5"
python gene_fake_data.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --gan_arch "SAGAN" \
    --gan_gene_ch 32 --gan_disc_ch 32 \
    --gan_embed_x2y_epoch 10 \
    --gan_niters 40000 --gan_resume_niters 0 --gan_save_niters_freq 5000 \
    --gan_d_niters 2 --gan_lr_g 1e-4 --gan_lr_d 1e-4 \
    --gan_batch_size_disc 100 --gan_batch_size_gene 100 --gan_batch_size_vis 10 \
    --num_grad_acc_d 1 --num_grad_acc_g 1 \
    --gan_threshold_type "soft" --gan_kappa $KAPPA \
    --gan_DiffAugment \
    --samp_batch_size 200 --samp_burnin_size 200 \
    --samp_num_fake_labels 0 --samp_nfake_per_label 200 \
    --subsampling \
    --dre_precnn_name resnet50 --dre_precnn_epochs 200 \
    --dre_epochs 100 --dre_lambda 0.05 \
    --filter --adjust --samp_filter_mae_percentile_threshold 0.7 \
    --samp_filter_precnn_net $net_t_name --samp_filter_precnn_net_ckpt_path $net_t_path \
    --unfiltered_fake_dataset_filename $unfiltered_fake_dataset_path \
    2>&1 | tee output_gene_SAGAN_subsampling_filter_adjust.txt