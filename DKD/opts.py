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