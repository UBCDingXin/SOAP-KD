import os
import timeit
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

## normalize images
def normalize_images(batch_images):
    batch_images = batch_images/255.0
    batch_images = (batch_images - 0.5)/0.5
    return batch_images



''' function for cnn training '''
def train_kd(net_t, net_s, reg_s, net_t_name, net_s_name, train_images, train_labels, testloader, epochs, resume_epoch=0, save_freq=40, batch_size=128, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[80, 150], weight_decay=1e-4, alpha=1.0, beta=1.0, path_to_ckpt = None, fn_denorm_labels=None):
    
    '''
    train_images: unnormalized images
    train_labels: normalized labels
    alpha: the coefficent for the hint loss
    beta: the coefficient for the RKD loss
    '''
    
    assert train_images.max()>1 and train_images.max()<=255.0 and train_images.min()>=0
    assert train_labels.min()>=0 and train_labels.max()<=1.0
    
    unique_train_labels = np.sort(np.array(list(set(train_labels)))) ##sorted unique labels
    
    indx_all = np.arange(len(train_labels))

    ''' learning rate decay '''
    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    net_t = net_t.cuda()
    net_s = net_s.cuda()
    if reg_s is not None:
        reg_s = reg_s.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net_s)
    if reg_s is not None:
        trainable_list.append(reg_s)

    optimizer = torch.optim.SGD(trainable_list.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    criterion_reg = nn.L1Loss()
    criterion_hint = nn.MSELoss() 

    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/S_{}_T_{}_a_{}_b_{}_checkpoint_epoch_{}.pth".format(net_s_name, net_t_name, alpha, beta, resume_epoch)
        checkpoint = torch.load(save_file)
        net_s.load_state_dict(checkpoint['net_s_state_dict'])
        if reg_s is not None:
            reg_s.load_state_dict(checkpoint['reg_s_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if


    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net_t.eval()
        net_s.train()
        train_loss = 0
        train_loss_reg = 0
        train_loss_hint = 0
        train_loss_rkd = 0
        
        adjust_learning_rate(optimizer, epoch)
        
        for batch_idx in range(len(train_labels)//batch_size):
            
            batch_train_indx = np.random.choice(indx_all, size=batch_size, replace=True).reshape(-1)
            
            ### get some real images for training
            batch_train_images = train_images[batch_train_indx]
            batch_train_images = normalize_images(batch_train_images) ## normalize real images
            batch_train_images = torch.from_numpy(batch_train_images).type(torch.float).cuda()
            assert batch_train_images.max().item()<=1.0
            batch_train_labels = train_labels[batch_train_indx]
            batch_train_labels = torch.from_numpy(batch_train_labels).type(torch.float).view(-1,1).cuda()
        
            #Forward pass
            outputs, feat_s = net_s(batch_train_images)
            _, feat_t = net_t(batch_train_images)

            loss1 = criterion_reg(outputs, batch_train_labels)
            loss2 = criterion_hint(feat_s.view(feat_s.size(0), -1), feat_t.view(feat_t.size(0), -1))
            
            # create pairs
            feat_s_diff = torch.repeat_interleave(feat_s, batch_size, dim=0) - feat_s.repeat(batch_size,1)
            feat_t_diff = torch.repeat_interleave(feat_t, batch_size, dim=0) - feat_t.repeat(batch_size,1)
            loss3 = torch.sum((feat_s_diff-feat_t_diff)**2)/(batch_size*(batch_size-1))

            indx_ones = torch.zeros(batch_size**2, 1)+1e-5
            for i in range(batch_size):
                indx_ones[i*batch_size] = 1
            indx_ones = indx_ones.type(torch.float).cuda()

            feat_dot_prod = torch.sum(feat_s_diff * feat_t_diff, dim=1)
            feat_norm_prod = torch.sqrt(torch.sum(feat_s_diff**2, dim=1)+indx_ones) * torch.sqrt(torch.sum(feat_t_diff**2, dim=1)+indx_ones)

            loss4 = torch.sum(1 - feat_dot_prod/feat_norm_prod) / (batch_size*(batch_size-1))

            loss = loss1 + loss2 + loss3 + loss4
            
            train_loss_reg += loss.cpu().item()

        
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        
        #end for batch_idx
        train_loss = train_loss / (len(train_labels)//batch_size)
        train_loss_reg = train_loss_reg / (len(train_labels)//batch_size)
        train_loss_hint = train_loss_hint / (len(train_labels)//batch_size)
        train_loss_rkd = train_loss_rkd / (len(train_labels)//batch_size)
        
        test_mae = test_cnn(net_s, testloader, fn_denorm_labels=fn_denorm_labels, verbose=False)
        
        print("S_{}_T_{}_a_{}_b_{}: [epoch {}/{}] train_loss:{:.2f} [{:.2f}/{:.2f}/{:.2f}], test_mae:{:.3f} Time:{:.2f}".format(net_s_name, net_t_name, alpha, beta, epoch+1, epochs, train_loss, train_loss_reg, train_loss_hint, train_loss_rkd, test_mae, timeit.default_timer()-start_time))

        # save checkpoint
        if path_to_ckpt is not None and ((epoch+1) % save_freq == 0 or (epoch+1) == epochs) :
            save_file = path_to_ckpt + "/S_{}_T_{}_a_{}_b_{}_checkpoint_epoch_{}.pth".format(net_s_name, net_t_name, alpha, beta, epoch+1)
            if reg_s is not None:
                torch.save({
                        'net_s_state_dict': net_s.state_dict(),
                        'reg_s_state_dict': reg_s.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'rng_state': torch.get_rng_state()
                }, save_file)
            else:
                torch.save({
                        'net_s_state_dict': net_s.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'rng_state': torch.get_rng_state()
                }, save_file)
    #end for epoch

    return net_s



def test_cnn(net, testloader, fn_denorm_labels=None, verbose=False):

    net = net.cuda()
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        abs_diff_avg = 0
        total = 0
        for _, (images, labels) in enumerate(testloader):
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.float).view(-1).cpu().numpy()
            outputs, _ = net(images)
            outputs = outputs.view(-1).cpu().numpy()
            labels = fn_denorm_labels(labels)
            outputs = fn_denorm_labels(outputs)
            abs_diff_avg += np.sum(np.abs(labels-outputs))
            total += len(labels)

    test_mae = abs_diff_avg/total
    if verbose:
        print('\n Test MAE: {}.'.format(test_mae))
    return test_mae