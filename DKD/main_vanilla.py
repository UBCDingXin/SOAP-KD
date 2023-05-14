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
from opts import cnn_opts
from models import *
from utils import *
from train_fn_cnn import train_cnn, test_cnn



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = cnn_opts()
print(args)

# os.environ['TORCH_HOME']=args.torch_model_path


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
output_directory = os.path.join(args.root_path, 'output/CNN/vanilla')
os.makedirs(output_directory, exist_ok=True)


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

## info of training set and test set
print("\n Training set: {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3], test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3]))



#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization
if args.cnn_name in ["vgg8"]:
    net = vgg8().cuda()
elif args.cnn_name in ["wrn_40_1", "wrn_40_2", "wrn_16_1", "wrn_16_2", "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet8x4", "resnet32x4"]:
    command_exec = "net = {}().cuda()".format(args.cnn_name)
    exec(command_exec)
else:
    net = model_builder(model_name=args.cnn_name, pretrained=args.pretrained).cuda()
# print(net)
num_parameters = count_parameters(net)

if not args.do_CV:
    filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_pretrain_{}_last.pth'.format(args.cnn_name, args.epochs, args.pretrained))
else:
    filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_pretrain_{}_last_inCV.pth'.format(args.cnn_name, args.epochs, args.pretrained))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training >>>")

    path_to_ckpt_in_train = output_directory + '/ckpts_in_train'  
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    train_cnn(net=net, net_name=args.cnn_name, train_images=train_images, train_labels=train_labels, testloader=testloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=args.save_freq, batch_size=args.batch_size_train, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, path_to_ckpt = path_to_ckpt_in_train, fn_denorm_labels=fn_denorm_labels)

    # store model
    torch.save({
        'net_state_dict': net.state_dict(),
    }, filename_ckpt)
    print("\n End training CNN.")
else:
    print("\n Loading pre-trained model.")
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
#end if

# testing
test_mae = test_cnn(net, testloader, fn_denorm_labels=fn_denorm_labels, verbose=True)
print("\n Test MAE {}.".format(test_mae))


if not args.do_CV:
    test_results_logging_fullpath = output_directory + '/test_results_{}_epoch_{}_pretrain_{}_MAE_{:.3f}.txt'.format(args.cnn_name, args.epochs, args.pretrained, test_mae)
else:
    test_results_logging_fullpath = output_directory + '/test_results_{}_epoch_{}_pretrain_{}_MAE_{:.3f}_inCV.txt'.format(args.cnn_name, args.epochs, args.pretrained, test_mae)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n {}; num paras: {}; seed: {} \n".format(args.cnn_name, num_parameters, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Test MAE {}.".format(test_mae))






print("\n===================================================================================================")