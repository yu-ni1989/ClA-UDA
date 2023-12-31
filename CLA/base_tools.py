#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torch.nn as nn
from scipy.spatial.distance import cdist
import numpy as np
from torch.autograd import grad
from itertools import chain
import random


def init_weights_orthogonal(m):
    if type(m) == nn.Conv2d:
        nn.init.orthogonal_(m.weight)
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)

def init_weights_xavier_normal(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def get_parameters(model):
    params = {}  # change the tpye of 'generator' into dict
    count = 0
    for name, param in model.named_parameters():
        params[count] = param
        count = count + 1
    return params

def discrepancy_2_params(paras1, paras2):
    length = len(paras1)
    loss = 0.0
    for i in range(length):
        loss_cur = torch.mean(torch.abs(paras1[i] - paras2[i]))
        loss = loss + loss_cur
    return loss


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1)- F.softmax(out2, dim=1)))

def discrepancy_l2(out1, out2):
    return torch.mean(torch.pow(F.softmax(out1, dim=1)- F.softmax(out2, dim=1), exponent=2.0))

def discrepancy_kl(out1, out2):
    p = F.softmax(out1, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(out1, dim=-1)
                                  - F.log_softmax(out2, dim=-1)), 1)
    return torch.mean(_kl)

def discrepancy_matrix(out1, out2):
    out1 = F.softmax(out1,dim=1)
    out2 = F.softmax(out2,dim=1)
    mul = out1.transpose(0, 1).mm(out2)
    loss_dis = torch.sum(mul) - torch.trace(mul)
    return loss_dis

def neg_exponential(x, par=1.0):
    return torch.exp(-x/par)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

# pseudo labels                                      
def obtain_label(loader, netE, netC1, netC2, args, c=None):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indexs = data[2]
            inputs = inputs.cuda()
            feas = netE(inputs)
            outputs1 = netC1(feas)
            outputs2 = netC2(feas)
            outputs = outputs1 + outputs2 
            #torch.stack([outputs1,outputs2]).mean(dim=0)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    #print("all_label:",all_label.size()[0],"right:",torch.squeeze(predict).float().eq(all_label.data).sum().item())
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    return pred_label.astype('int')


def gradient_discrepancy_loss(args, preds_s1,preds_s2, src_y, preds_t1, preds_t2, tgt_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()
    #CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)
    total_loss = 0
    c_candidate = list(range(args.class_num))
    random.shuffle(c_candidate)
    # gmn iterations
    for c in c_candidate[0:args.gmn_N]:
        # gm loss
        gm_loss = 0

        src_ind = (src_y == c).nonzero().squeeze()
        #print("src_y,",src_y,"src_ind:",src_ind)
        tgt_ind = (tgt_y == c).nonzero().squeeze()
        if src_ind.shape == torch.Size([]) or tgt_ind.shape == torch.Size([]) or src_ind.shape == torch.Size([0]) or tgt_ind.shape == torch.Size([0]):
            continue

        p_s1 = preds_s1[src_ind]
        p_s2 = preds_s2[src_ind]
        p_t1 = preds_t1[tgt_ind]
        p_t2 = preds_t2[tgt_ind]
        s_y = src_y[src_ind]
        t_y = tgt_y[tgt_ind]
        
        #print("src_ind:",s_y,"tgt_ind:",t_y)

        src_loss1 = loss(p_s1 , s_y)
        
        tgt_loss1 = loss_w(p_t1 , t_y)

        src_loss2 = loss(p_s2 , s_y)
        tgt_loss2 = loss_w(p_t2 , t_y)

        grad_cossim11 = []
        #grad_mse11 = []
        grad_cossim22 = []
        #grad_mse22 = []

        #netE+C1
        for n, p in netC1.named_parameters():
            # if len(p.shape) == 1: continue

            real_grad = grad([src_loss1],
                                [p],
                                create_graph=True,
                                only_inputs=True,
                                allow_unused=False)[0]
            fake_grad = grad([tgt_loss1],
                                [p],
                                create_graph=True,
                                only_inputs=True,
                                allow_unused=False)[0]

            if len(p.shape) > 1:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
            else:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
            #_mse = F.mse_loss(fake_grad, real_grad)
            grad_cossim11.append(_cossim)
            #grad_mse.append(_mse)

        grad_cossim1 = torch.stack(grad_cossim11)
        gm_loss1 = (1.0 - grad_cossim1).sum()
        #grad_mse1 = torch.stack(grad_mse)
        #gm_loss1 = (1.0 - grad_cossim1).sum() * args.Q + grad_mse1.sum() * args.Z
        #netE+C2
        for n, p in netC2.named_parameters():
            # if len(p.shape) == 1: continue

            real_grad = grad([src_loss2],
                                [p],
                                create_graph=True,
                                only_inputs=True)[0]
            fake_grad = grad([tgt_loss2],
                                [p],
                                create_graph=True,
                                only_inputs=True)[0]

            if len(p.shape) > 1:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
            else:
                _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
            #_mse = F.mse_loss(fake_grad, real_grad)
            grad_cossim22.append(_cossim)
            #grad_mse.append(_mse)

        grad_cossim2 = torch.stack(grad_cossim22)
        #grad_mse2 = torch.stack(grad_mse)
        #gm_loss2 = (1.0 - grad_cossim2).sum() * args.Q + grad_mse2.sum() * args.Z
        gm_loss2 = (1.0 - grad_cossim2).sum()
        gm_loss = (gm_loss1 + gm_loss2)/2.0
        total_loss += gm_loss
        
    return total_loss/args.gmn_N

def gradient_discrepancy_loss_margin(args, p_s1,p_s2, s_y, p_t1, p_t2, t_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()
    #CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)
    # gm loss
    gm_loss = 0

    #print("src_ind:",s_y,"tgt_ind:",t_y)

    src_loss1 = loss(p_s1 , s_y)
    
    tgt_loss1 = loss_w(p_t1 , t_y)

    src_loss2 = loss(p_s2 , s_y)
    tgt_loss2 = loss_w(p_t2 , t_y)

    grad_cossim11 = []
    #grad_mse11 = []
    grad_cossim22 = []
    #grad_mse22 = []

    #netE+C1
    for n, p in netC1.named_parameters():
        # if len(p.shape) == 1: continue

        real_grad = grad([src_loss1],
                            [p],
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=False)[0]
        fake_grad = grad([tgt_loss1],
                            [p],
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=False)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        #_mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim11.append(_cossim)
        #grad_mse.append(_mse)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss1 = (1.0 - grad_cossim1).mean()
    #grad_mse1 = torch.stack(grad_mse)
    #gm_loss1 = (1.0 - grad_cossim1).sum() * args.Q + grad_mse1.sum() * args.Z
    #netE+C2
    for n, p in netC2.named_parameters():
        # if len(p.shape) == 1: continue

        real_grad = grad([src_loss2],
                            [p],
                            create_graph=True,
                            only_inputs=True)[0]
        fake_grad = grad([tgt_loss2],
                            [p],
                            create_graph=True,
                            only_inputs=True)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        #_mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim22.append(_cossim)
        #grad_mse.append(_mse)

    grad_cossim2 = torch.stack(grad_cossim22)
    #grad_mse2 = torch.stack(grad_mse)
    #gm_loss2 = (1.0 - grad_cossim2).sum() * args.Q + grad_mse2.sum() * args.Z
    gm_loss2 = (1.0 - grad_cossim2).mean()
    gm_loss = (gm_loss1 + gm_loss2)/2.0
        
    return gm_loss



def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy 

def Entropy_condition(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy 

def Entropy_inf(input_):
    return Entropy_condition(input_) + Entropy_div(input_)

def Weighted_CrossEntropy(input_,labels):
    input_s = F.softmax(input_)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    #print("cross:",nn.CrossEntropyLoss(reduction='none')(input_, labels))
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))


def GetWeightLossFormTwoClassifiers(F1, F2):
    W5 = None
    W6 = None

    for (w5, w6) in zip(F1.parameters(), F2.parameters()):
        if W5 is None and W6 is None:
            W5 = w5.view(-1)
            W6 = w6.view(-1)
        else:
            W5 = torch.cat((W5, w5.view(-1)), 0)
            W6 = torch.cat((W6, w6.view(-1)), 0)

    loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
    return loss_weight


def GetLabelsUsingTwoPredictors(pred1, pred2):
    values1, indices1 = pred1.data.max(1)
    values2, indices2 = pred2.data.max(1)
    ind_b = values2 > values1
    indices1[ind_b] = indices2[ind_b]
    return indices1

def GetPredUsingTwoPredictors(pred1, pred2):
    pred = pred1
    values1, indices1 = pred1.data.max(1)
    values2, indices2 = pred2.data.max(1)
    ind_b = values2 > values1
    ind_b = ind_b.repeat(5, 1, 1).unsqueeze(0)
    pred[ind_b] = pred2[ind_b]
    return pred
