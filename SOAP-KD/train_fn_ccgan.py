import torch
import numpy as np
import os
import timeit
from PIL import Image
from torchvision.utils import save_image


from utils import *
from opts import gen_synth_data_opts
from DiffAugment_pytorch import DiffAugment

''' Settings '''
args = gen_synth_data_opts()


# some parameters in opts
loss_type = args.gan_loss_type
niters = args.gan_niters
resume_niters = args.gan_resume_niters
d_niters = args.gan_d_niters
dim_gan = args.gan_dim_g
lr_g = args.gan_lr_g
lr_d = args.gan_lr_d
save_niters_freq = args.gan_save_niters_freq
batch_size_disc = args.gan_batch_size_disc
batch_size_gene = args.gan_batch_size_gene
batch_size_max = max(batch_size_disc, batch_size_gene)

batch_size_vis = args.gan_batch_size_vis ## batch size for visualization

threshold_type = args.gan_threshold_type
nonzero_soft_weight_threshold = args.gan_nonzero_soft_weight_threshold

use_DiffAugment = args.gan_DiffAugment
policy = args.gan_DiffAugment_policy

## 梯度累积
num_grad_acc_d = args.num_grad_acc_d
num_grad_acc_g = args.num_grad_acc_g


def train_ccgan(kernel_sigma, kappa, train_images, train_labels, netG, netD, net_y2h, save_images_folder, path_to_ckpt = None, clip_label=False, use_amp=True):

    '''
    Note that train_images are not normalized to [-1,1]
    train_labels are normalized to [0,1]
    '''

    assert train_images.max()>1.0 and train_images.min()>=0 and train_images.max()<=255.0
    assert train_labels.min()>=0 and train_labels.max()<=1.0
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    netG = netG.cuda()
    netD = netD.cuda()
    net_y2h = net_y2h.cuda()
    net_y2h.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # # 混合精度
    # # 在训练最开始之前实例化一个GradScaler对象
    # scaler = torch.cuda.amp.GradScaler()


    if path_to_ckpt is not None and resume_niters>0:
        save_file = path_to_ckpt + "/CcGAN_checkpoint_niters_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if
   

    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    n_row=10; n_col = n_row
    z_fixed = torch.randn(n_row*n_col, dim_gan, dtype=torch.float).cuda()
    start_label = np.quantile(train_labels, 0.05)
    end_label = np.quantile(train_labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row*n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i*n_col+j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).cuda()
    

    z_fixed = torch.cat((z_fixed, z_fixed[0:batch_size_vis]), dim=0)
    y_fixed = torch.cat((y_fixed, y_fixed[0:batch_size_vis]), dim=0)


    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        for d_i in range(d_niters):
            '''  Train Discriminator   '''
            ## randomly draw batch_size_disc y's from unique_train_labels
            batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_disc, replace=True)
            ## add Gaussian noise; we estimate image distribution conditional on these labels
            batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_disc)
            batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
            if clip_label:
                batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)

            ## find index of real images with labels in the vicinity of batch_target_labels
            ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
            batch_real_indx = np.zeros(batch_size_disc, dtype=int) #index of images in the data; the labels of these images are in the vicinity
            batch_fake_labels = np.zeros(batch_size_disc)
            batch_size_of_vicinity = torch.zeros(batch_size_disc)

            for j in range(batch_size_disc):
                ## index for real images
                if threshold_type == "hard":
                    indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                else:
                    # reverse the weight function for SVDL
                    indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

                ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
                while len(indx_real_in_vicinity)<1:
                    batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                    batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                    if clip_label:
                        batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                    # index for real images
                    if threshold_type == "hard":
                        indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                    else:
                        # reverse the weight function for SVDL
                        indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]
                #end while len(indx_real_in_vicinity)<1

                assert len(indx_real_in_vicinity)>=1
                batch_size_of_vicinity[j] = len(indx_real_in_vicinity)

                batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

                ## labels for fake images generation
                if threshold_type == "hard":
                    lb = batch_target_labels[j] - kappa
                    ub = batch_target_labels[j] + kappa
                else:
                    lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                    ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                lb = max(0.0, lb); ub = min(ub, 1.0)
                assert lb<=ub
                assert lb>=0 and ub>=0
                assert lb<=1 and ub<=1
                batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
            #end for j

            ## draw the real image batch from the training set
            batch_real_images = train_images[batch_real_indx]
            assert batch_real_images.max()>1
            batch_real_labels = train_labels[batch_real_indx]
            batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).cuda()


            ## normalize real images
            trainset = IMGs_dataset(batch_real_images, labels=None, normalize=True)
            train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_disc, shuffle=False)
            train_dataloader = iter(train_dataloader)
            batch_real_images = train_dataloader.next()
            assert len(batch_real_images) == batch_size_disc
            batch_real_images = batch_real_images.type(torch.float).cuda()
            assert batch_real_images.max().item()<=1


            ## generate the fake image batch
            batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).cuda()
            z = torch.randn(batch_size_disc, dim_gan, dtype=torch.float).cuda()
            batch_fake_images = netG(z, net_y2h(batch_fake_labels))

            ## target labels on gpu
            batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()

            ## weight vector
            if threshold_type == "soft":
                real_weights = torch.exp(-kappa*(batch_real_labels-batch_target_labels)**2).cuda()
                fake_weights = torch.exp(-kappa*(batch_fake_labels-batch_target_labels)**2).cuda()
            else:
                real_weights = torch.ones(batch_size_disc, dtype=torch.float).cuda()
                fake_weights = torch.ones(batch_size_disc, dtype=torch.float).cuda()

            # forward pass
            if use_DiffAugment:
                real_dis_out = netD(DiffAugment(batch_real_images, policy=policy), net_y2h(batch_target_labels))
                fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=policy), net_y2h(batch_target_labels))
            else:
                real_dis_out = netD(batch_real_images, net_y2h(batch_target_labels))
                fake_dis_out = netD(batch_fake_images.detach(), net_y2h(batch_target_labels))

            if loss_type == "vanilla":
                real_dis_out = torch.nn.Sigmoid()(real_dis_out)
                fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
                d_loss_real = - torch.log(real_dis_out+1e-20)
                d_loss_fake = - torch.log(1-fake_dis_out+1e-20)
            elif loss_type == "hinge":
                d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
                d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)

            d_loss = torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) + torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1))

            # d_loss.backward()
            # if ((d_i*niter+1)%num_grad_acc_d)==0:
            #     optimizerD.step()
            #     optimizerD.zero_grad()

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()
            
        ##end D update


        '''  Train Generator   '''
        netG.train()

        # generate fake images
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_gene, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_gene)
        batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
        if clip_label:
            batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).cuda()

        z = torch.randn(batch_size_gene, dim_gan, dtype=torch.float).cuda()
        batch_fake_images = netG(z, net_y2h(batch_target_labels))

        # loss
        if use_DiffAugment:
            dis_out = netD(DiffAugment(batch_fake_images, policy=policy), net_y2h(batch_target_labels))
        else:
            dis_out = netD(batch_fake_images, net_y2h(batch_target_labels))
        if loss_type == "vanilla":
            dis_out = torch.nn.Sigmoid()(dis_out)
            g_loss = - torch.mean(torch.log(dis_out+1e-20))
        elif loss_type == "hinge":
            g_loss = - dis_out.mean()

        ## backward
        # g_loss.backward()
        # if ((niter+1)%num_grad_acc_g)==0:
        #     optimizerG.step()
        #     optimizerG.zero_grad()
        
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        # print loss
        if (niter+1) % 20 == 0:
            print ("CcGAN: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_time))

        if (niter+1) % 500 == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = []
                n_got = 0
                while n_got<n_row*n_col:
                    gen_imgs_tmp = netG(z_fixed[n_got:(n_got+batch_size_vis)], net_y2h(y_fixed[n_got:(n_got+batch_size_vis)]))
                    gen_imgs_tmp = gen_imgs_tmp.detach().cpu()
                    gen_imgs.append(gen_imgs_tmp)
                    n_got += len(gen_imgs_tmp)
                gen_imgs = torch.cat(gen_imgs, dim=0)
                gen_imgs = gen_imgs[0:n_row*n_col]
                
                save_image(gen_imgs.data, save_images_folder + '/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if path_to_ckpt is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = path_to_ckpt + "/CcGAN_checkpoint_niters_{}.pth".format(niter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for niter
    return netG, netD


def SampCcGAN_given_labels(netG, net_y2h, labels, batch_size = 100, to_numpy=True, verbose=True):
    '''
    labels: a numpy array; normalized label in [0,1]
    '''

    assert labels.min()>=0 and labels.max()<=1.0


    nfake = len(labels)
    if batch_size>nfake:
        batch_size=nfake

    fake_images = []
    fake_labels = np.concatenate((labels, labels[0:batch_size]))
    
    netG = netG.cuda()
    netG.eval()
    net_y2h = net_y2h.cuda()
    net_y2h.eval()
    
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        n_img_got = 0
        while n_img_got < nfake:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
            y = torch.from_numpy(fake_labels[n_img_got:(n_img_got+batch_size)]).type(torch.float).view(-1,1).cuda()
            batch_fake_images = netG(z, net_y2h(y))
            fake_images.append(batch_fake_images.cpu())
            n_img_got += batch_size
            if verbose:
                pb.update(min(float(n_img_got)/nfake, 1)*100)
        ##end while

    fake_images = torch.cat(fake_images, dim=0)
    #remove extra entries
    fake_images = fake_images[0:nfake]
    fake_labels = fake_labels[0:nfake]

    if to_numpy:
        fake_images = fake_images.numpy()
        
    netG = netG.cpu()
    net_y2h = net_y2h.cpu()

    return fake_images, fake_labels