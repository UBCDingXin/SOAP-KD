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

    ''' CNN Settings '''
    parser.add_argument('--cnn_name', type=str, default='vgg16')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=50)
    parser.add_argument('--lr_base', type=float, default=0.01, help='base learning rate of CNNs')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='80_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    return args