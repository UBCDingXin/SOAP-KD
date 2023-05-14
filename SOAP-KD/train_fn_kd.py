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
    criterion_rkd = RKDLoss()

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
            # ### generate target labels
            # batch_target_labels = np.random.choice(unique_train_labels, size=batch_size, replace=True)
            # batch_unique_train_labels, batch_unique_label_counts = np.unique(batch_target_labels, return_counts=True)

            # batch_train_indx = []
            # for j in range(len(batch_unique_train_labels)):
            #     indx_j = np.where(train_labels==batch_unique_train_labels[j])[0]
            #     indx_j = np.random.choice(indx_j, size=batch_unique_label_counts[j])
            #     batch_train_indx.append(indx_j)
            # batch_train_indx = np.concatenate(batch_train_indx)
            # batch_train_indx = batch_train_indx.reshape(-1)
            
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
            loss = criterion_reg(outputs, batch_train_labels)
            
            train_loss_reg += loss.cpu().item()

            if alpha>0 or beta>0:
                if reg_s is not None:
                    feat_s = reg_s(feat_s)
                _, feat_t = net_t(batch_train_images)
                
                if alpha>0:
                    loss_hint = criterion_hint(feat_s.view(feat_s.size(0), -1), feat_t.view(feat_t.size(0), -1))
                    loss = loss + alpha*loss_hint
                    train_loss_hint += loss_hint.cpu().item()
                    
                if beta>0:
                    loss_rkd = criterion_rkd(feat_s.view(feat_s.size(0), -1), feat_t.view(feat_t.size(0), -1))
                    loss = loss + beta*loss_rkd
                    train_loss_rkd += loss_rkd.cpu().item()
        
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



class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
