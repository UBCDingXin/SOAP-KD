print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm, trange
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image

### import my stuffs ###
from opts import gen_synth_data_opts
from utils import SimpleProgressBar, IMGs_dataset, PlotLoss, count_parameters
from models import *
from train_fn_ccgan import train_ccgan, SampCcGAN_given_labels
from train_fn_cnn_embed import train_net_embed, train_net_y2h
from train_fn_cnn_extract import train_cnn_extract
from train_fn_cdre import train_cdre


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)

os.environ['TORCH_HOME']=args.torch_model_path

if args.subsampling:
    subsampling_method = "cDR-RS_precnn_{}_epochs_{}_DR_{}_epochs_{}_lambda_{:.3f}".format(args.dre_precnn_name, args.dre_precnn_epochs, args.dre_net, args.dre_epochs, args.dre_lambda)
else:
    subsampling_method = "None"

## filter??
if args.filter:
    subsampling_method = subsampling_method + "_filter_{}_perc_{:.2f}".format(args.samp_filter_precnn_net, args.samp_filter_mae_percentile_threshold)
else:
    subsampling_method = subsampling_method + "_filter_None"

## adjust labels??
subsampling_method = subsampling_method + "_adjust_{}".format(args.adjust)

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
output_directory = os.path.join(args.root_path, 'output')
os.makedirs(output_directory, exist_ok=True)
## folders for CcGAN and cDRE and fake data
save_models_folder = os.path.join(output_directory, 'CcGAN/saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(output_directory, 'CcGAN/saved_images')
os.makedirs(save_images_folder, exist_ok=True)
fake_data_folder = os.path.join(output_directory, 'fake_data')
os.makedirs(fake_data_folder, exist_ok=True)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
print('\n Loading real data...')
print(args.data_path)

## for original train/test split
hf = h5py.File(args.data_path, 'r')
images_train = hf['train_images'][:]
labels_train = hf['train_orientation'][:]
images_test = hf['test_images'][:]
labels_test = hf['test_orientation'][:]
cls_labels_train = hf['train_cls_labels'][:]
cls_labels_test = hf['test_cls_labels'][:]
hf.close()

# ## for resplit
# hf = h5py.File(args.data_path, 'r')
# train_indx = hf['train_indx'][:]
# test_indx = hf['test_indx'][:]
# images_train = hf['all_images'][train_indx]
# labels_train = hf['all_orientation'][train_indx]
# images_test = hf['all_images'][test_indx]
# labels_test = hf['all_orientation'][test_indx]
# hf.close()

print("\r Data loaded...")

min_label = 0
max_label = 180.0
assert labels_train.min()>= min_label and labels_test.min()>=min_label
assert labels_train.max()<= max_label and labels_test.max()<=max_label
assert len(list(set(cls_labels_train)))==23 and len(list(set(cls_labels_test)))==23

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


# unique normalized training labels
## normalize training labels to [0,1]
labels_train_norm = fn_norm_labels(labels_train)
unique_labels_train_norm = np.sort(np.array(list(set(labels_train_norm))))
labels_test_norm = fn_norm_labels(labels_test)


## set sigma and kappa/nu in CcGAN
if args.gan_kernel_sigma<0:
    std_label = np.std(labels_train_norm)
    args.gan_kernel_sigma = 1.06*std_label*(len(labels_train_norm))**(-1/5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print("\n The std of {} normalied training labels is {} so the kernel sigma is {}".format(len(labels_train_norm), std_label, args.gan_kernel_sigma))
##end if

if args.gan_kappa<0:
    n_unique = len(unique_labels_train_norm)

    diff_list = []
    for i in range(1,n_unique):
        diff_list.append(unique_labels_train_norm[i] - unique_labels_train_norm[i-1])
    kappa_base = np.abs(args.gan_kappa)*np.max(np.array(diff_list))

    print("\n The maximum gap of the normalized training labels is {}".format(max(diff_list)))

    if args.gan_threshold_type=="hard":
        args.gan_kappa = kappa_base
    else:
        args.gan_kappa = 1/kappa_base**2
## end if




#######################################################################################
'''                         Pre-trained CNN  for label embedding                    '''
#######################################################################################

net_embed_x2y_filename_ckpt = save_models_folder + '/ckpt_embed_{}_epoch_{}_seed_{}.pth'.format(args.gan_embed_x2y_net_name, args.gan_embed_x2y_epoch, args.seed)
print(net_embed_x2y_filename_ckpt)
net_embed_y2h_filename_ckpt = save_models_folder + '/ckpt_embed_y2h_epoch_{}_seed_{}.pth'.format(args.gan_embed_y2h_epoch, args.seed)
print(net_embed_y2h_filename_ckpt)

testset_embed_x2y = IMGs_dataset(images_test, labels_test_norm, normalize=True)
testloader_embed_x2y = torch.utils.data.DataLoader(testset_embed_x2y, batch_size=100, shuffle=False, num_workers=args.num_workers)

net_embed_x2y = vgg8_embed(dim_embed=args.gan_dim_embed)

net_embed_x2y = net_embed_x2y.cuda()
net_embed_x2y = nn.DataParallel(net_embed_x2y)

net_embed_y2h = model_y2h(dim_embed=args.gan_dim_embed)
net_embed_y2h = net_embed_y2h.cuda()
net_embed_y2h = nn.DataParallel(net_embed_y2h)

## (1). Train net_embed first: x2h+h2y
if not os.path.isfile(net_embed_x2y_filename_ckpt):
    print("\n Start training CNN for label embedding >>>")

    # lr decay epochs
    net_embed_x2y_lr_decay_epochs = (args.gan_embed_x2y_lr_decay_epochs).split("_")
    net_embed_x2y_lr_decay_epochs = [int(epoch) for epoch in net_embed_x2y_lr_decay_epochs]

    # ckpts in training
    ckpts_in_train_net_embed_x2y = os.path.join(save_models_folder, 'ckpts_in_train_embed_x2y_{}'.format(args.gan_embed_x2y_net_name))
    os.makedirs(ckpts_in_train_net_embed_x2y, exist_ok=True)
    
    net_test_mae_file_fullpath = os.path.join(ckpts_in_train_net_embed_x2y, 'test_mae_embed_{}_epoch_{}_seed_{}.png'.format(args.gan_embed_x2y_net_name, args.gan_embed_x2y_epoch, args.seed))

    # training function
    net_embed_x2y, train_mse_all, test_mae_all = train_net_embed(net=net_embed_x2y, train_images=images_train, train_labels=labels_train_norm, test_loader=testloader_embed_x2y, epochs=args.gan_embed_x2y_epoch, resume_epoch=args.gan_embed_x2y_resume_epoch, save_freq=40, batch_size=args.gan_embed_x2y_batch_size, lr_base=args.gan_embed_x2y_lr_base, lr_decay_factor=args.gan_embed_x2y_lr_decay_factor, lr_decay_epochs=net_embed_x2y_lr_decay_epochs, weight_decay=1e-4, path_to_ckpt = ckpts_in_train_net_embed_x2y, fn_denorm_labels=fn_denorm_labels)

    PlotLoss(loss=test_mae_all, filename=net_test_mae_file_fullpath)

    # save model
    torch.save({
    'net_state_dict': net_embed_x2y.state_dict(),
    }, net_embed_x2y_filename_ckpt)
else:
    print("\n net_embed ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_x2y_filename_ckpt)
    net_embed_x2y.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

## (2). Train y2h
#train a net which maps a label back to the embedding space
if not os.path.isfile(net_embed_y2h_filename_ckpt):
    print("\n Start training net_embed_y2h >>>")

    # lr decay epochs
    net_embed_y2h_lr_decay_epochs = (args.gan_embed_y2h_lr_decay_epochs).split("_")
    net_embed_y2h_lr_decay_epochs = [int(epoch) for epoch in net_embed_y2h_lr_decay_epochs]

    # training function
    net_embed_y2h = train_net_y2h(unique_train_labels_norm=unique_labels_train_norm, net_y2h=net_embed_y2h, net_embed=net_embed_x2y, epochs=args.gan_embed_y2h_epoch, lr_base=args.gan_embed_y2h_lr_base, lr_decay_factor=args.gan_embed_y2h_lr_decay_factor, lr_decay_epochs=net_embed_y2h_lr_decay_epochs, weight_decay=1e-4, batch_size=args.gan_embed_y2h_batch_size)

    # save model
    torch.save({
    'net_state_dict': net_embed_y2h.state_dict(),
    }, net_embed_y2h_filename_ckpt)
else:
    print("\n net_embed_y2h ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(net_embed_y2h_filename_ckpt)
    net_embed_y2h.load_state_dict(checkpoint['net_state_dict'])
#end not os.path.isfile

##some simple test after the embedding nets training
indx_tmp = np.arange(len(unique_labels_train_norm))
np.random.shuffle(indx_tmp)
indx_tmp = indx_tmp[:10]
labels_tmp = unique_labels_train_norm[indx_tmp].reshape(-1,1)
labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).cuda()
epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).cuda()
labels_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
net_embed_x2y.eval()
net_embed_h2y = net_embed_x2y.module.h2y
net_embed_y2h.eval()
with torch.no_grad():
    labels_rec_tmp = net_embed_h2y(net_embed_y2h(labels_tmp)).cpu().numpy().reshape(-1,1)
results = np.concatenate((labels_tmp.cpu().numpy(), labels_rec_tmp), axis=1)
print("\n labels vs reconstructed labels")
print(results)





#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("\n Start CcGAN training: {}, Sigma is {:.3f}, Kappa is {:.3f}".format(args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa))

path_to_ckpt_ccgan = os.path.join(save_models_folder, 'ckpt_{}_loss_{}_niters_{}_seed_{}_{}_sigma{:.3f}_kappa{:.3f}.pth'.format(args.gan_arch, args.gan_loss_type, args.gan_niters, args.seed, args.gan_threshold_type, args.gan_kernel_sigma, args.gan_kappa))
print(path_to_ckpt_ccgan)

start = timeit.default_timer()
if not os.path.isfile(path_to_ckpt_ccgan):
    ## images generated during training
    images_in_train_ccgan = os.path.join(save_images_folder, 'images_in_train_{}'.format(args.gan_arch))
    os.makedirs(images_in_train_ccgan, exist_ok=True)

    # ckpts in training
    ckpts_in_train_ccgan = os.path.join(save_models_folder, 'ckpts_in_train_{}'.format(args.gan_arch))
    os.makedirs(ckpts_in_train_ccgan, exist_ok=True)

    # init models
    if args.gan_arch == 'SAGAN':
        netG = SAGAN_Generator(z_dim=args.gan_dim_g, gene_ch=args.gan_gene_ch, dim_embed=args.gan_dim_embed).cuda()
        netD = SAGAN_Discriminator(disc_ch=args.gan_disc_ch, dim_embed=args.gan_dim_embed).cuda()
    elif args.gan_arch == 'SNGAN':
        netG = SNGAN_Generator(z_dim=args.gan_dim_g, gene_ch=args.gan_gene_ch, dim_embed=args.gan_dim_embed).cuda()
        netD = SNGAN_Discriminator(disc_ch=args.gan_disc_ch, dim_embed=args.gan_dim_embed).cuda()
    else:
        raise Exception('Wrong CcGAN name!')
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # training function
    netG, netD = train_ccgan(kernel_sigma=args.gan_kernel_sigma, kappa=args.gan_kappa, train_images=images_train, train_labels=labels_train_norm, netG=netG, netD=netD, net_y2h = net_embed_y2h, save_images_folder = images_in_train_ccgan, path_to_ckpt = ckpts_in_train_ccgan, clip_label=False)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, path_to_ckpt_ccgan)

else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(path_to_ckpt_ccgan)
    if args.gan_arch == 'SAGAN':
        netG = SAGAN_Generator(z_dim=args.gan_dim_g, gene_ch=args.gan_gene_ch, dim_embed=args.gan_dim_embed).cuda()
    elif args.gan_arch == 'SNGAN':
        netG = SNGAN_Generator(z_dim=args.gan_dim_g, gene_ch=args.gan_gene_ch, dim_embed=args.gan_dim_embed).cuda()
    else:
        raise Exception('Wrong CcGAN name!')
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])
## end if
stop = timeit.default_timer()
print("CcGAN training finished; Time elapses: {}s".format(stop - start))


def fn_sampleGAN_given_label(nfake, given_label, netG=netG, net_y2h=net_embed_y2h, batch_size = 100, to_numpy=True, denorm=True, verbose=False):
    ''' label: normalized label in [0,1] '''
    ''' output labels are still normalized '''
    
    assert 0<=given_label<=1.0
    
    netG = netG.cuda()
    net_y2h = net_y2h.cuda()
    netG.eval()
    net_y2h.eval()

    if batch_size>nfake:
        batch_size = nfake
    
    fake_images = []
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        n_img_got = 0
        while n_img_got < nfake:
            y = np.ones(batch_size) * given_label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
            z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
            batch_fake_images = netG(z, net_y2h(y))
            if denorm:
                batch_fake_images = (batch_fake_images*0.5+0.5)*255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
            fake_images.append(batch_fake_images.cpu())
            n_img_got += len(batch_fake_images)
            if verbose:
                pb.update(min(float(n_img_got)/nfake, 1)*100)
    fake_images = torch.cat(fake_images, dim=0)
    fake_labels = torch.ones(nfake) * given_label #use assigned label

    if to_numpy:
        fake_images = fake_images.numpy()
        fake_labels = fake_labels.numpy()

    netG = netG.cpu()
    net_y2h = net_y2h.cpu()

    return fake_images[0:nfake], fake_labels[0:nfake]



#######################################################################################
'''                                    cDRE training                                 '''
#######################################################################################

if args.subsampling:
    ##############################################
    ''' Pre-trained CNN for feature extraction '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n Pre-trained CNN for feature extraction")
    
    # filename
    filename_precnn_ckpt = save_models_folder + '/ckpt_PreCNNForDRE_{}_epoch_{}_seed_{}.pth'.format(args.dre_precnn_name, args.dre_precnn_epochs, args.seed)
    print('\n ' + filename_precnn_ckpt)

    # initialize net
    if args.dre_precnn_name in ["vgg8"]:
        dre_precnn_extract_net = vgg8(num_class=23).cuda()
    else:
        dre_precnn_extract_net = dre_extractor_builder(model_name=args.dre_precnn_name, num_class=23, pretrained=True).cuda()
    dre_precnn_extract_net = nn.DataParallel(dre_precnn_extract_net)
    count_parameters(dre_precnn_extract_net)

    # training
    if not os.path.isfile(filename_precnn_ckpt):

        testset_extract = IMGs_dataset(images_test, cls_labels_test, normalize=True)
        testloader_extract = torch.utils.data.DataLoader(testset_extract, batch_size=100, shuffle=False, num_workers=args.num_workers)

        ckpts_in_train_precnn = os.path.join(save_models_folder, 'ckpts_in_train_extractor_{}'.format(args.dre_precnn_name))
        os.makedirs(ckpts_in_train_precnn, exist_ok=True)

        lr_decay_epochs = (args.dre_precnn_lr_decay_freq).split("_")
        lr_decay_epochs = [int(epoch) for epoch in lr_decay_epochs]

        print("\n Start training model for feature extraction in the DRE >>>")
        train_cnn_extract(net=dre_precnn_extract_net, net_name=args.dre_precnn_name, train_images=images_train, train_labels=cls_labels_train, testloader=testloader_extract, epochs=args.dre_precnn_epochs, resume_epoch=args.dre_precnn_resume_epoch, save_freq=50, batch_size=args.dre_precnn_batch_size_train, lr_base=args.dre_precnn_lr_base, lr_decay_factor=args.dre_precnn_lr_decay_factor, lr_decay_epochs=lr_decay_epochs, weight_decay=args.dre_precnn_weight_decay, path_to_ckpt = ckpts_in_train_precnn)

        # store model
        torch.save({
            'net': dre_precnn_extract_net.state_dict(),
        }, filename_precnn_ckpt)
        print("\n End training CNN.")
    else:
        print("\n Loading pre-trained VGG8 for feature extraction in DRE.")
        checkpoint = torch.load(filename_precnn_ckpt)
        dre_precnn_extract_net.load_state_dict(checkpoint['net'])
    #end if



    ##############################################
    ''' cDRE Training '''
    print("\n -----------------------------------------------------------------------------------------")
    print("\n cDRE training")

    ### dr model filename
    drefile_fullpath = save_models_folder + "/ckpt_cDR-RS_{}_precnn_{}_epochs_{}_DR_{}_lambda_{:.3f}_epochs_{}_seed_{}.pth".format(args.gan_arch, args.dre_precnn_name, args.dre_precnn_epochs, args.dre_net, args.dre_lambda, args.dre_epochs, args.seed)
    print('\n' + drefile_fullpath)

    path_to_ckpt_in_train = save_models_folder + '/ckpt_cDR-RS_{}_precnn_{}_epochs_{}_DR_{}_lambda_{:.3f}_seed_{}'.format(args.gan_arch, args.dre_precnn_name, args.dre_precnn_epochs, args.dre_net, args.dre_lambda, args.seed)
    os.makedirs(path_to_ckpt_in_train, exist_ok=True)

    dre_loss_file_fullpath = save_models_folder + '/train_loss_cDR-RS_{}_precnn_{}_epochs_{}_DR_{}_epochs_{}_lambda_{:.3f}_seed_{}.png'.format(args.gan_arch, args.dre_precnn_name, args.dre_precnn_epochs, args.dre_net, args.dre_epochs, args.dre_lambda, args.seed)

    ### dre training
    dre_net = cDR_CNN(img_size=args.img_size, dim_cond=args.gan_dim_embed)
        
    num_parameters_DR = count_parameters(dre_net)
    dre_net = nn.DataParallel(dre_net)
    #if DR model exists, then load the pretrained model; otherwise, start training the model.
    if not os.path.isfile(drefile_fullpath):
        print("\n Begin Training conditional DR in Feature Space: >>>")
        dre_net, avg_train_loss = train_cdre(unique_labels=unique_labels_train_norm, train_images=images_train, train_labels=labels_train_norm, dre_net=dre_net, dre_precnn_net=dre_precnn_extract_net, netG=netG, net_y2h=net_embed_y2h, path_to_ckpt=path_to_ckpt_in_train)

        # save model
        torch.save({
        'net_state_dict': dre_net.state_dict(),
        }, drefile_fullpath)
        PlotLoss(avg_train_loss, dre_loss_file_fullpath)
    else:
        # if already trained, load pre-trained DR model
        checkpoint_dre_net = torch.load(drefile_fullpath)
        dre_net.load_state_dict(checkpoint_dre_net['net_state_dict'])
    ##end if not

    # Compute density ratio: function for computing a bunch of images in a numpy array
    def comp_cond_density_ratio(imgs, labels, dre_precnn_net=dre_precnn_extract_net, dre_net=dre_net, net_y2h=net_embed_y2h, batch_size=args.samp_batch_size):

        assert imgs.max()>1
        assert labels.min()>=0 and labels.max()<=1.0

        dre_precnn_net = dre_precnn_net.cuda()
        dre_net = dre_net.cuda()
        net_y2h = net_y2h.cuda()
        dre_precnn_net.eval()
        dre_net.eval()
        net_y2h.eval()

        #imgs: a torch tensor
        n_imgs = len(imgs)
        if batch_size>n_imgs:
            batch_size = n_imgs

        assert imgs.max().item()>1.0 ##make sure all images are not normalized
        assert labels.max()<=1.0 and labels.min()>=0 ##make sure all labels are normalized to [0,1]

        ##make sure the last iteration has enough samples
        imgs = torch.cat((imgs, imgs[0:batch_size]), dim=0)
        labels = torch.cat((labels, labels[0:batch_size]), dim=0)

        density_ratios = []
        # print("\n Begin computing density ratio for images >>")
        with torch.no_grad():
            n_imgs_got = 0
            while n_imgs_got < n_imgs:
                batch_images = imgs[n_imgs_got:(n_imgs_got+batch_size)]
                batch_images = (batch_images/255.0-0.5)/0.5 ## normalize
                batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
                batch_images = batch_images.type(torch.float).cuda()
                batch_labels = batch_labels.type(torch.float).view(-1,1).cuda()
                batch_labels = net_y2h(batch_labels)
                _, batch_features = dre_precnn_net(batch_images)
                batch_ratios = dre_net(batch_features, batch_labels)
                density_ratios.append(batch_ratios.cpu().detach())
                n_imgs_got += batch_size
            ### while n_imgs_got
        density_ratios = torch.cat(density_ratios)
        density_ratios = density_ratios[0:n_imgs].numpy()
        return density_ratios

    # Enhanced sampler based on the trained DR model
    # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
    def fn_enhancedSampler_given_label(nfake, given_label, batch_size=args.samp_batch_size, verbose=True):
        ''' given_label is normalized '''
        assert 0<=given_label<=1.0
        
        ## Burn-in Stage
        n_burnin = args.samp_burnin_size
        burnin_imgs, burnin_labels = fn_sampleGAN_given_label(n_burnin, given_label, batch_size = batch_size, to_numpy=False, denorm=True)
        burnin_densityratios = comp_cond_density_ratio(burnin_imgs, burnin_labels)
        # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
        M_bar = np.max(burnin_densityratios)
        del burnin_imgs, burnin_densityratios; gc.collect()
        ## Rejection sampling
        enhanced_imgs = []
        if verbose:
            pb = SimpleProgressBar()
            # pbar = tqdm(total=nfake)
        num_imgs = 0
        while num_imgs < nfake:
            batch_imgs, batch_labels = fn_sampleGAN_given_label(batch_size, given_label, batch_size = batch_size, to_numpy=False, denorm=True)
            batch_ratios = comp_cond_density_ratio(batch_imgs, batch_labels)
            batch_imgs = batch_imgs.numpy() #convert to numpy array
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #threshold
            batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where(batch_psi<=batch_p)[0]
            if len(indx_accept)>0:
                enhanced_imgs.append(batch_imgs[indx_accept])
            num_imgs+=len(indx_accept)
            del batch_imgs, batch_ratios; gc.collect()
            if verbose:
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
                # pbar.update(len(indx_accept))
        # pbar.close()
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        return enhanced_imgs, given_label*np.ones(nfake)





#######################################################################################
'''                                   Sampling                                      '''
#######################################################################################

#--------------------------------------------------------------------------------------
''' Synthetic Data Generation '''

print('\n Start sampling ...')


dump_fake_images_filename = os.path.join(fake_data_folder, 'FGSC23_fake_images_{}_{}_Nlabel_{}_NFakePerLabel_{}_seed_{}.h5'.format(args.gan_arch, subsampling_method, args.samp_num_fake_labels, args.samp_nfake_per_label, args.seed))
print(dump_fake_images_filename)

if not os.path.isfile(dump_fake_images_filename):

    fake_data_h5file_fullpath = os.path.join(fake_data_folder, args.unfiltered_fake_dataset_filename)
    print(fake_data_h5file_fullpath)
    if os.path.isfile(fake_data_h5file_fullpath) and (args.filter or args.adjust):
        print("\n Loading exiting unfiltered fake data >>>")
        hf = h5py.File(fake_data_h5file_fullpath, 'r')
        fake_images = hf['fake_images'][:]
        fake_labels = hf['fake_labels'][:] #unnormalized
        hf.close()
    else:
        if fake_data_h5file_fullpath!="None":
            print("\r Assigned unfilered fake dataset does not exist!!!")
        
        if args.samp_num_fake_labels>0:
            target_labels_norm = np.union1d(np.linspace(0.0, 1.0, args.samp_num_fake_labels), np.arange(181)/max_label)
        else:  
            target_labels_norm = unique_labels_train_norm        

        if args.subsampling:
            print("\n Generating {} fake images for each of {} distinct labels with subsampling: {}.".format(args.samp_nfake_per_label, len(target_labels_norm), subsampling_method))
            fake_images = []
            fake_labels = []
            for i in trange(len(target_labels_norm)):
                fake_images_i, fake_labels_i = fn_enhancedSampler_given_label(args.samp_nfake_per_label, target_labels_norm[i], batch_size=args.samp_batch_size, verbose=False)
                ### append
                fake_images.append(fake_images_i)
                fake_labels.append(fake_labels_i)
            ##end for i
        else:
            print("\n Generating {} fake images for each of {} distinct labels without subsampling.".format(args.samp_nfake_per_label, len(target_labels_norm)))
            fake_images = []
            fake_labels = []
            for i in trange(len(target_labels_norm)):
                fake_images_i, fake_labels_i = fn_sampleGAN_given_label(nfake=args.samp_nfake_per_label, given_label=target_labels_norm[i], batch_size=args.samp_batch_size, verbose=False)
                ### append
                fake_images.append(fake_images_i)
                fake_labels.append(fake_labels_i)
            ##end for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels, axis=0)
        ### denormalize labels
        fake_labels = np.clip(fake_labels, 0, 1)
        fake_labels = fn_denorm_labels(fake_labels)
        
        assert len(fake_images) == args.samp_nfake_per_label*len(target_labels_norm)
        assert len(fake_labels) == args.samp_nfake_per_label*len(target_labels_norm)
        assert fake_images.max()>1
        assert fake_labels.max()>1
        ##end if
    ##end if os


    #--------------------------------------------------------------------------------------
    ''' Filtered and Adjusted by a pre-trained CNN '''
    if args.filter or args.adjust:
        print("\n -----------------------------------------------------------------------------------------")
        print("\n Start Filtering Synthetic Data >>>")

        ## dataset
        assert fake_images.max()>1
        assert fake_labels.max()>1
        dataset_filtering = IMGs_dataset(fake_images, fake_labels, normalize=True)
        dataloader_filtering = torch.utils.data.DataLoader(dataset_filtering, batch_size=args.samp_filter_batch_size, shuffle=False, num_workers=args.num_workers)

        ## load pre-trained cnn
        if args.samp_filter_precnn_net in ["vgg8"]:
            filter_precnn_net = vgg8().cuda()
        elif args.samp_filter_precnn_net in ["wrn_40_1", "wrn_40_2", "wrn_16_1", "wrn_16_2", "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet8x4", "resnet32x4"]:
            command_exec = "filter_precnn_net = {}().cuda()".format(args.samp_filter_precnn_net)
            exec(command_exec)
        else:
            filter_precnn_net = model_builder(model_name=args.samp_filter_precnn_net).cuda()
        # print(net)
        num_parameters = count_parameters(filter_precnn_net)
        filter_precnn_net = filter_precnn_net.cuda()
        checkpoint = torch.load(args.samp_filter_precnn_net_ckpt_path)
        filter_precnn_net.load_state_dict(checkpoint['net_state_dict'])

        ## evaluate on fake data
        fake_mae_loss = []
        fake_labels_pred = []
        filter_precnn_net.eval()
        pbar = tqdm(total=len(fake_images))
        with torch.no_grad():
            total = 0
            loss_all = 0
            for batch_idx, (images, labels) in enumerate(dataloader_filtering):
                images = images.type(torch.float).cuda()
                labels = labels.type(torch.float) #unnormalized label
                labels_pred, _ = filter_precnn_net(images)
                labels_pred = torch.clip(labels_pred, 0, 1)
                labels_pred = fn_denorm_labels(labels_pred.cpu()) #denormalize
                labels = labels.view(-1)
                labels_pred = labels_pred.view(-1)
                loss = torch.abs(labels_pred-labels)
                loss_all += loss.sum().item()
                total += labels.size(0)
                fake_labels_pred.append(labels_pred.numpy())
                fake_mae_loss.append(loss.numpy())
                pbar.update(len(images))
            print('\n Test MAE of {} on the {} fake images: {}.'.format(args.samp_filter_precnn_net, len(fake_images), loss_all / total))
        fake_mae_loss = np.concatenate(fake_mae_loss, axis=0)
        fake_labels_pred = np.concatenate(fake_labels_pred, axis=0)

        mae_cutoff_point = np.quantile(fake_mae_loss, q=args.samp_filter_mae_percentile_threshold)
        indx_sel = np.where(fake_mae_loss<mae_cutoff_point)[0]
        fake_images = fake_images[indx_sel]
        if args.adjust:
            fake_labels = fake_labels_pred[indx_sel] #adjust the labels of fake data by using the pre-trained CNN
        else:
            fake_labels = fake_labels[indx_sel]

        ## histogram of MAEs
        fig = plt.figure()
        ax = plt.subplot(111)
        n, bins, patches = plt.hist(fake_mae_loss, 100, density=False, facecolor='g', alpha=0.75)
        plt.axvline(x=mae_cutoff_point, c='grey')
        plt.xlabel('MAE')
        plt.ylabel('Frequency')
        plt.title('Histogram of MAE')
        plt.grid(True)
        #plt.show()
        plt.savefig(os.path.join(fake_data_folder, 'histogram_of_fake_data_MAE_with_subsampling_{}_MAEFilter_{}.png'.format(subsampling_method, args.samp_filter_mae_percentile_threshold)))

    #--------------------------------------------------------------------------------------
    ''' Dump synthetic data to h5 file '''
    with h5py.File(dump_fake_images_filename, "w") as f:
        f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
        f.create_dataset('fake_labels', data = fake_labels, dtype='float')


else:
    print('\n Start loading generated fake data...')
    with h5py.File(dump_fake_images_filename, "r") as f:
        fake_images = f['fake_images'][:]
        fake_labels = f['fake_labels'][:]


# if not os.path.isfile(dump_fake_images_filename):
#     with h5py.File(dump_fake_images_filename, "w") as f:
#         f.create_dataset('fake_images', data = fake_images, dtype='uint8', compression="gzip", compression_opts=6)
#         f.create_dataset('fake_labels', data = fake_labels, dtype='float')
# else:
#     print('\n Start loading generated fake data...')
#     with h5py.File(dump_fake_images_filename, "r") as f:
#         fake_images = f['fake_images'][:]
#         fake_labels = f['fake_labels'][:]
        
print("\n The dim of the fake dataset: ", fake_images.shape)
print("\n The range of generated fake dataset: MIN={}, MAX={}.".format(fake_labels.min(), fake_labels.max()))



### visualize data distribution
unique_fake_labels = np.sort(np.array(list(set(fake_labels))))
frequencies = []
for i in range(len(unique_fake_labels)):
    indx_i = np.where(fake_labels==unique_fake_labels[i])[0]
    frequencies.append(len(indx_i))
frequencies = np.array(frequencies).astype(int)
# plot data in grouped manner of bar type
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.grid(color='lightgrey', linestyle='--', zorder=0)
ax.bar(unique_fake_labels, frequencies, align='center', color='tab:green', zorder=3)
ax.set_xlabel("Angle")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(fake_data_folder, "FGSC23_fake_images_{}_{}_Nlabel_{}_NFakePerLabel_{}_data_dist.pdf".format(args.gan_arch, subsampling_method, args.samp_num_fake_labels, args.samp_nfake_per_label)))
plt.close()

print('\n Frequence of angles: MIN={}, MEAN={}, MAX={}, SUM={}.'.format(np.min(frequencies),np.mean(frequencies),np.max(frequencies),np.sum(frequencies)))



print("\n===================================================================================================")
