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
import copy

### import my stuffs ###
from opts import cnn_opts
from model import vgg
from utils import *
from train_fn import train_cnn, test_cnn



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
output_directory = os.path.join(args.root_path, 'output/ncls_{}'.format(args.num_classes))
os.makedirs(output_directory, exist_ok=True)


#######################################################################################
'''                                Data loader                                      '''
#######################################################################################
print('\n Loading training data ...')
hf = h5py.File(args.data_path, 'r')
train_images = hf['train_images'][:]
train_labels = hf['train_orientation'][:]
test_images = hf['test_images'][:]
test_labels = hf['test_orientation'][:]
indx_subtrain = hf['indx_subtrain'][:]
indx_valid = hf['indx_valid'][:]
hf.close()

min_label = 0
max_label = 180
assert train_labels.min()>= min_label and test_labels.min()>=min_label
assert train_labels.max()<= max_label and test_labels.max()<=max_label
assert args.num_classes>0 and args.num_classes<=181


#treat SOAP as classification; convert angles to class labels
unique_labels = np.sort(np.array(list(set(train_labels))))
num_unique_labels = len(unique_labels)
print("{} unique labels are split into {} classes".format(num_unique_labels, args.num_classes))

if args.num_classes<181:
    ## convert angles to class labels and vice versa
    ### step 1: prepare two dictionaries
    label2class = dict()
    class2label = dict()
    num_labels_per_class = num_unique_labels//args.num_classes
    class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes; each interval is a class
    curr_class = 0
    for i in range(num_unique_labels):
        label2class[unique_labels[i]]=curr_class
        if (i+1)%num_labels_per_class==0 and (curr_class+1)!=args.num_classes:
            curr_class += 1
            class_cutoff_points.append(unique_labels[i+1])
    class_cutoff_points.append(unique_labels[-1])
    assert len(class_cutoff_points)-1 == args.num_classes

    ### the cell label of each interval equals to the average of the two end points
    for i in range(args.num_classes):
        class2label[i] = float((class_cutoff_points[i]+class_cutoff_points[i+1])/2)

    print(class_cutoff_points)

else:
    label2class = dict()
    class2label = dict()

    for i in range(num_unique_labels):
        label2class[unique_labels[i]]= int(i)

    for i in range(args.num_classes):
        class2label[i] = float(i)
## end if args.num_classes

### step 2: convert angles to class labels
train_labels_new = -1*np.ones(len(train_labels)).astype(int)
for i in range(len(train_labels)):
    train_labels_new[i] = label2class[train_labels[i]]
assert np.sum(train_labels_new<0)==0

unique_labels = np.sort(np.array(list(set(train_labels_new)))).astype(int)
assert args.num_classes == len(unique_labels)


## plot num of samples per class
nsamp_per_class = []
for i in range(args.num_classes):
    nsamp_per_class.append(len(np.where(train_labels_new==i)[0]))
nsamp_per_class = np.array(nsamp_per_class)

bar_chart_filename = os.path.join(output_directory, 'bar_chart_nsamp_per_class.png')
plt.figure()
plt.bar(np.arange(args.num_classes,dtype=int), nsamp_per_class, width=0.8, bottom=None)
plt.savefig(bar_chart_filename)




## number of real images
nreal = len(train_labels)
assert len(train_labels) == len(train_images)


## data loader for the training set and test set
assert test_labels.max()>1
testset = IMGs_dataset(test_images, test_labels, normalize=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

## info of training set and test set
print("\n Training set: {}x{}x{}x{}; Testing set: {}x{}x{}x{}.".format(train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3], test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3]))


#######################################################################################
'''                                  CNN Training                                   '''
#######################################################################################

### model initialization
net = vgg(args.cnn_name, pretrained=True, num_classes=args.num_classes).cuda()
# print(net)
num_parameters = count_parameters(net)


filename_ckpt = os.path.join(output_directory, 'ckpt_{}_epoch_{}_last.pth'.format(args.cnn_name, args.epochs))
print('\n' + filename_ckpt)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Start training >>>")

    path_to_ckpt_in_train = output_directory + '/ckpts_in_train'  
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    train_cnn(net=net, net_name=args.cnn_name, train_images=train_images, train_labels=train_labels_new, testloader=testloader, epochs=args.epochs, resume_epoch=args.resume_epoch, save_freq=args.save_freq, batch_size=args.batch_size_train, lr_base=args.lr_base, lr_decay_factor=args.lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.weight_decay, path_to_ckpt = path_to_ckpt_in_train, class2label=class2label)

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
test_mae = test_cnn(net, testloader, class2label=class2label, verbose=True)
print("\n Test MAE {}.".format(test_mae))



test_results_logging_fullpath = output_directory + '/test_results_{}_MAE_{:.3f}.txt'.format(args.cnn_name, test_mae)
if not os.path.isfile(test_results_logging_fullpath):
    test_results_logging_file = open(test_results_logging_fullpath, "w")
    test_results_logging_file.close()
with open(test_results_logging_fullpath, 'a') as test_results_logging_file:
    test_results_logging_file.write("\n===================================================================================================")
    test_results_logging_file.write("\n {}; num paras: {}; seed: {} \n".format(args.cnn_name, num_parameters, args.seed))
    print(args, file=test_results_logging_file)
    test_results_logging_file.write("\n Test MAE {}.".format(test_mae))






print("\n===================================================================================================")