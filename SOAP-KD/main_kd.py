print("\n===================================================================================================")

import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from torchvision.utils import save_image
import csv
from tqdm import tqdm
import gc
import h5py


### import my stuffs ###
from opts import kd_opts
from models import *
from utils import *
from train_fn_kd import train_kd, test_cnn


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = kd_opts()
print(args)

os.environ['TORCH_HOME']=args.torch_model_path


#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)


#-------------------------------
# CNN settings
## lr decay scheme
lr_decay_epochs = (args.lr_decay_epochs).split("_")
lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]


#-------------------------------
# output folders

if not args.use_fake_data:
    output_directory = os.path.join(args.root_path, 'output/CNN/kd_vanilla')
    os.makedirs(output_directory, exist_ok=True)
else:
    fake_data_name = args.fake_data_path.split('/')[-1]
    output_directory = os.path.join(args.root_path, 'output/CNN/kd_{}_useNfake_{}'.format(fake_data_name, args.nfake))


#######################################################################################
'''                                Data loader                                      '''
#######################################################################################
print('\n Loading training data ...')
## for original train/test split
hf = h5py.File(args.data_path, 'r')
train_images = hf['train_images'][:]
train_labels = hf['train_orientation'][:]
test_images = hf['test_images'][:]
test_labels = hf['test_orientation'][:]
indx_subtrain = hf['indx_subtrain'][:]
indx_valid = hf['indx_valid'][:]
hf.close()

if args.do_CV:
    print("\n In validation mode !!!")
    test_images = train_images[indx_valid]
    test_labels = train_labels[indx_valid]
    train_images = train_images[indx_subtrain]
    train_labels = train_labels[indx_subtrain]

min_label = 0
max_label = 180
assert train_labels.min()>= min_label and test_labels.min()>=min_label
assert train_labels.max()<= max_label and test_labels.max()<=max_label


# some functions
def fn_norm_labels(labels):
    '''
    labels: unnormalized labels
    '''
    shift_value = np.abs(min_label)
    labels_after_shift = labels + shift_value
    max_label_after_shift = max_label + shift_value
    
    return labels_after_shift/max_label_after_shift

def fn_denorm_labels(labels):
    '''
    labels: normalized labels; numpy array
    '''
    shift_value = np.abs(min_label)
    max_label_after_shift = max_label + shift_value
    labels = labels * max_label_after_shift
    labels = labels - shift_value
    
    return labels

## normalize to [0,1]
train_labels = fn_norm_labels(train_labels)
test_labels = fn_norm_labels(test_labels)

## number of real images
nreal = len(train_labels)
assert len(train_labels) == len(train_images)


## data loader for the training set and test set
testset = IMGs_dataset(test_images, test_labels, normalize=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)



## load fake dataset
if args.use_fake_data and args.fake_data_path != 'None':
    print("\n Start loading fake data: {}...".format(args.fake_data_path))
    hf = h5py.File(args.fake_data_path, 'r')
    fake_images = hf['fake_images'][:]
    fake_labels = hf['fake_labels'][:]
    hf.close()
    print('\n Fake images: {}, min {}, max {}.'.format(fake_images.shape, fake_images.min(), fake_images.max()))
    print('\n Fake labels: {}, min {}, max {}.'.format(fake_labels.shape, fake_labels.min(), fake_labels.max()))
    assert np.max(fake_images)>1 and np.min(fake_images)>=0

    if args.nfake>0 and args.nfake<len(fake_labels):
        indx_fake = np.arange(len(fake_labels))
        np.random.shuffle(indx_fake)
        indx_fake = indx_fake[0:int(args.nfake)]
        fake_images = fake_images[indx_fake]
        fake_labels = fake_labels[indx_fake]

    assert len(fake_images)==len(fake_labels)
    fake_labels = fn_norm_labels(fake_labels)
    fake_labels = np.clip(fake_labels, 0, 1)
    
    print("\n Range of normalized fake labels: ", fake_labels.min(), fake_labels.max())

    ## combine fake and real
    train_images = np.concatenate((train_images, fake_images), axis=0)
    train_labels = np.concatenate((train_labels, fake_labels))

    del fake_images, fake_labels
    gc.collect()


## info of training set and test set
print("\n Training set: {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3], test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3]))


#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization
print("\r Init. teacher model...")
if args.net_t_name in ["vgg8"]:
    net_t = vgg8().cuda()
elif args.net_t_name in ["wrn_40_1", "wrn_40_2", "wrn_16_1", "wrn_16_2", "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet8x4", "resnet32x4"]:
    command_exec = "net_t = {}().cuda()".format(args.net_t_name)
    exec(command_exec)
else:
    net_t = model_builder(model_name=args.net_t_name, pretrained=True).cuda()
num_parameters = count_parameters(net_t)

checkpoint_t = torch.load(args.net_t_path)
net_t.load_state_dict(checkpoint_t['net_state_dict'])

print("\r Init. student model...")
if args.net_s_name in ["vgg8"]:
    net_s = vgg8().cuda()
elif args.net_s_name in ["wrn_40_1", "wrn_40_2", "wrn_16_1", "wrn_16_2", "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet8x4", "resnet32x4"]:
    command_exec = "net_s = {}().cuda()".format(args.net_s_name)
    exec(command_exec)
else:
    net_s = model_builder(model_name=args.net_s_name, pretrained=True).cuda()
num_parameters = count_parameters(net_s)

if args.finetune_net_s and args.init_s_path != 'None':
    print("\r Load student ckpt...")
    checkpoint_s = torch.load(args.init_s_path)
    net_s.load_state_dict(checkpoint_s['net_s_state_dict'])


## regressor for hint
tmp_input = torch.randn(4,3,224,224).cuda()
_, feat_t = net_t(tmp_input)
_, feat_s = net_s(tmp_input)

if feat_t.view(feat_t.size(0), -1).size(1)==feat_s.view(feat_s.size(0), -1).size(1):
    reg_s = None
else:
    reg_s = hint_regressor(args.net_t_name, args.net_s_name).cuda()

if not args.do_CV:
    filename_ckpt = os.path.join(output_directory, 'ckpt_S_{}_T_{}_a_{}_b_{}_epoch_{}_last.pth'.format(args.net_s_name, args.net_t_name, args.alpha, args.beta, args.epochs))
else:
    filename_ckpt = os.path.join(output_directory, 'ckpt_S_{}_T_{}_a_{}_b_{}_epoch_{}_last_inCV.pth'.format(args.net_s_name, args.net_t_name, args.alpha, args.beta, args.epochs))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training >>>")

    path_to_ckpt_in_train = output_directory + '/ckpts_in_train'  
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    train_kd(net_t=net_t, net_s=net_s, reg_s=reg_s, net_t_name=args.net_t_name, net_s_name=args.net_s_name, train_images=train_images, train_labels=train_labels, testloader=testloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=args.save_freq, batch_size=args.batch_size_train, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, alpha=args.alpha, beta=args.beta, path_to_ckpt = path_to_ckpt_in_train, fn_denorm_labels=fn_denorm_labels)

    # store model
    torch.save({
        'net_s_state_dict': net_s.state_dict(),
    }, filename_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained model.")
    checkpoint = torch.load(filename_ckpt)
    net_s.load_state_dict(checkpoint['net_s_state_dict'])
#end if


# testing
test_mae = test_cnn(net_s, testloader, fn_denorm_labels=fn_denorm_labels, verbose=True)


if not args.do_CV:
    test_results_logging_fullpath = output_directory + '/test_results_S_{}_T_{}_a_{}_b_{}_epoch_{}_MAE_{:.3f}.txt'.format(args.net_s_name, args.net_t_name, args.alpha, args.beta, args.epochs, test_mae)
else:
    test_results_logging_fullpath = output_directory + '/test_results_S_{}_T_{}_a_{}_b_{}_epoch_{}_MAE_{:.3f}_inCV.txt'.format(args.net_s_name, args.net_t_name, args.alpha, args.beta, args.epochs, test_mae)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n S_{}_T_{}_a_{}_b_{}; num paras: {}; seed: {} \n".format(args.net_s_name, args.net_t_name, args.alpha, args.beta, num_parameters, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Test MAE {}.".format(test_mae))






print("\n===================================================================================================")