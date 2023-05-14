import argparse



'''

Options for Some Baseline CNN Training

'''
def cnn_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--torch_model_path', type=str, default='')
    
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--do_CV', action='store_true', default=False) #validation
    # parser.add_argument('--nfake', type=int, default=0, help='number of fake images for training')
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=224, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--cnn_name', type=str, default='vgg16')
    parser.add_argument('--pretrained', action='store_true', default=False)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='80_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--init_model_path', type=str, default='')

    args = parser.parse_args()

    return args



'''

Options for GAN, DRE, and synthetic data generation

'''
def gen_synth_data_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--torch_model_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed (default: 2023)')
    parser.add_argument('--num_workers', type=int, default=0)
    
    ''' Dataset '''
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=224, metavar='N')

    ''' GAN settings '''
    # label embedding setting
    parser.add_argument('--gan_dim_embed', type=int, default=128) #dimension of the embedding space

    parser.add_argument('--gan_embed_x2y_net_name', type=str, default='vgg8')
    parser.add_argument('--gan_embed_x2y_epoch', type=int, default=200) #epoch of cnn training for label embedding
    parser.add_argument('--gan_embed_x2y_resume_epoch', type=int, default=0) #epoch of cnn training for label embedding
    parser.add_argument('--gan_embed_x2y_batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--gan_embed_x2y_lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--gan_embed_x2y_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--gan_embed_x2y_lr_decay_epochs', type=str, default='60_120', help='decay lr at which epoch; separate by _')

    parser.add_argument('--gan_embed_y2h_epoch', type=int, default=500)
    parser.add_argument('--gan_embed_y2h_batch_size', type=int, default=128, metavar='N')
    parser.add_argument('--gan_embed_y2h_lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--gan_embed_y2h_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--gan_embed_y2h_lr_decay_epochs', type=str, default='80_150', help='decay lr at which epoch; separate by _')

    # gan setting
    parser.add_argument('--gan_arch', type=str, default='SAGAN')
    parser.add_argument('--gan_loss_type', type=str, default='hinge')
    parser.add_argument('--gan_niters', type=int, default=20000, help='number of iterations')
    parser.add_argument('--gan_resume_niters', type=int, default=0)
    parser.add_argument('--gan_save_niters_freq', type=int, default=5000, help='frequency of saving checkpoints')
    parser.add_argument('--gan_d_niters', type=int, default=1, help='update D multiple times while update G once')
    parser.add_argument('--gan_lr_g', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--gan_lr_d', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--gan_dim_g', type=int, default=256, help='Latent dimension of GAN')
    parser.add_argument('--gan_batch_size_disc', type=int, default=128)
    parser.add_argument('--gan_batch_size_gene', type=int, default=128)
    parser.add_argument('--gan_gene_ch', type=int, default=64)
    parser.add_argument('--gan_disc_ch', type=int, default=64)
    parser.add_argument('--gan_batch_size_vis', type=int, default=100)

    # ccgan setting
    parser.add_argument('--gan_kernel_sigma', type=float, default=-1.0,
                        help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
    parser.add_argument('--gan_threshold_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--gan_kappa', type=float, default=-5.0)
    parser.add_argument('--gan_nonzero_soft_weight_threshold', type=float, default=1e-3,
                        help='threshold for determining nonzero weights for SVDL; we neglect images with too small weights')

    # DiffAugment setting
    parser.add_argument('--gan_DiffAugment', action='store_true', default=False)
    parser.add_argument('--gan_DiffAugment_policy', type=str, default='color,translation,cutout')

    # Gradient accumulation
    parser.add_argument('--num_grad_acc_d', type=int, default=1)
    parser.add_argument('--num_grad_acc_g', type=int, default=1)


    ''' DRE Settings '''
    ## Pre-trained CNN for feature extraction
    parser.add_argument('--dre_precnn_name', type=str, default='vgg16')
    parser.add_argument('--dre_precnn_epochs', type=int, default=200)
    parser.add_argument('--dre_precnn_resume_epoch', type=int, default=0, metavar='N')
    parser.add_argument('--dre_precnn_lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--dre_precnn_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_precnn_lr_decay_freq', type=str, default='80_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_precnn_batch_size_train', type=int, default=128, metavar='N')
    parser.add_argument('--dre_precnn_weight_decay', type=float, default=1e-4)
    
    ## DR model in the feature space
    parser.add_argument('--dre_net', type=str, default='CNN5', help='DR Model in the feature space') # DRE in Feature Space
    parser.add_argument('--dre_epochs', type=int, default=100)
    parser.add_argument('--dre_resume_epoch', type=int, default=0)
    parser.add_argument('--dre_save_freq', type=int, default=50)
    parser.add_argument('--dre_lr_base', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dre_lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--dre_lr_decay_epochs', type=str, default='30_60', help='decay lr at which epoch; separate by _')
    parser.add_argument('--dre_batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training DRE')
    parser.add_argument('--dre_lambda', type=float, default=0.05, help='penalty in DRE')
    parser.add_argument('--dre_optimizer', type=str, default='ADAM')


    ''' Sampling Settings '''
    parser.add_argument('--subsampling', action='store_true', default=False, help='cDR-RS based subsampling')
    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--adjust', action='store_true', default=False)
    
    parser.add_argument('--samp_batch_size', type=int, default=100) #also used for computing density ratios after the dre training
    parser.add_argument('--samp_burnin_size', type=int, default=100)
    parser.add_argument('--samp_num_fake_labels', type=int, default=500)
    parser.add_argument('--samp_nfake_per_label', type=int, default=100)
    parser.add_argument('--samp_filter_precnn_net', type=str, default='vgg19',
                        help='Pre-trained CNN for filtering and label adjustment;')
    parser.add_argument('--samp_filter_precnn_net_ckpt_path', type=str, default='')
    parser.add_argument('--samp_filter_mae_percentile_threshold', type=float, default=1.0,
                        help='The percentile threshold of MAE to filter out bad synthetic images by a pre-trained net')
    parser.add_argument('--unfiltered_fake_dataset_filename', type=str, default='')
    parser.add_argument('--samp_filter_batch_size', type=int, default=100)
    
    args = parser.parse_args()

    return args




'''

Options for knowledge distillation

'''
def kd_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--torch_model_path', type=str, default='')
    
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--do_CV', action='store_true', default=False) #validation
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=224, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--net_t_name', type=str, default='vgg16')
    parser.add_argument('--net_t_path', type=str, default='')
    parser.add_argument('--net_s_name', type=str, default='resnet8')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='80_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    ''' KD Settings '''
    parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for hint loss')
    parser.add_argument('--beta', type=float, default=1.0, help='weight balance for RKD loss')
    
    parser.add_argument('--use_fake_data', action='store_true', default=False)
    parser.add_argument('--fake_data_path', type=str, default='None')
    parser.add_argument('--nfake', type=int, default=0)
    parser.add_argument('--finetune_net_s', action='store_true', default=False)
    parser.add_argument('--init_s_path', type=str, default='None')

    args = parser.parse_args()

    return args