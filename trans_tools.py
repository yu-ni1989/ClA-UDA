#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import numpy as np
import torch
import torch.nn as nn
from DA.CLA.opt_tools import *

def ass_vector2matrix(ass_ind_1, ass_ind_2, length):
    ass_matrix = np.zeros([length, length])
    for r, c in zip(ass_ind_1, ass_ind_2):
        ass_matrix[r, c] = 1.0

    return ass_matrix

def vector2matrix(ass_ind_1, ass_ind_2, weights, length):
    ass_matrix = np.zeros([length, length])
    weight_matrix = np.zeros([length, length])
    count = 0
    for r, c in zip(ass_ind_1, ass_ind_2):
        ass_matrix[r, c] = 1.0
        weight_matrix[r, c] = weights[count]
        count = count + 1

    return ass_matrix, weight_matrix

def label_2_onehot(label, num_classes):
    onehot_out = F.one_hot(label.data[0].long(), num_classes)
    onehot_out = onehot_out.unsqueeze(0).permute(0, 3, 1, 2)
    return onehot_out

def labels_transform_pool(label_s, label_t, num_classes, scale=32):
    image_shape = label_s.shape
    # interp_pred = nn.Upsample(size=(image_shape[-2]//scale, image_shape[-1]//scale), mode='bilinear')
    down_raw = nn.AvgPool2d((scale*2, scale*2), padding=scale - 1, stride=scale)
    down_o = nn.AvgPool2d((scale//4, scale//4), padding=scale//8 - 1, stride=scale//8)

    label_s_temp = label_s.clone()
    label_s_temp[label_s >= num_classes] = num_classes
    label_s_temp[label_s < 0] = num_classes
    out_s = label_2_onehot(label_s_temp, num_classes+1).float()
    out_s = down_raw(out_s)

    # out_t = interp_pred(label_t)
    out_t = down_o(label_t)
    zero_pad = torch.zeros(1, 1, out_t.shape[-2], out_t.shape[-1]).to(label_t.device)
    out_t = torch.cat((out_t, zero_pad), dim=1)
    out_label_s = out_s.view(image_shape[0], num_classes+1, -1).transpose(2, 1)
    out_label_t = out_t.view(image_shape[0], num_classes+1, -1).transpose(2, 1)

    return out_label_s, out_label_t

def image_dmm_lab_nonst(dmm, image_s, image_t, label_s, label_t, num_classes, scale=32, wflag=False, pflag=False):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_pool(label_s, label_t, num_classes, scale=scale)

    ass_matrix = None
    weights_out = None
    trans_matrix = None
    if wflag:
        ass_ind_1, ass_ind_2, weights, weights_out = assignment_for_arrays_cosine(label_s_vec, F.softmax(label_t_vec, dim=2), wflag=wflag,
                                                     pflag=pflag)
        ass_matrix, weight_matrix = vector2matrix(ass_ind_1, ass_ind_2, weights, length=temp_size[0] * temp_size[1])
        trans_matrix, tflag = dmm.get_transferring_matrix(image_s_vec.detach().cpu().numpy(),
                                                   image_t_vec.detach().cpu().numpy(),
                                                   ass_ind_1, ass_ind_2, weights=weights)
        ass_matrix = torch.tensor(ass_matrix).float().detach()
        weights_out = torch.tensor(weights_out).float().detach()
        if tflag == False:
            return image_s, ass_matrix, weights_out
    else:
        ass_ind_1, ass_ind_2 = assignment_for_arrays_cosine(label_s_vec, F.softmax(label_t_vec, dim=2), wflag=wflag, pflag=pflag)
        ass_matrix = ass_vector2matrix(ass_ind_1, ass_ind_2, length=temp_size[0] * temp_size[1])
        trans_matrix, tflag = dmm.get_transferring_matrix(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(),
                                               ass_ind_1, ass_ind_2, None)
        ass_matrix = torch.tensor(ass_matrix).float().detach()
        if tflag == False:
            return image_s, ass_matrix, weights_out
    trans_matrix = torch.tensor(trans_matrix).float()

    image_s_temp = image_s.view(input_size_source[0], input_size_source[1], -1).transpose(2, 1)
    image_t_temp = image_t.view(input_size_target[0], input_size_target[1], -1).transpose(2, 1)

    means_s = torch.mean(image_s_temp, dim=1).unsqueeze(1).squeeze(0)
    means_t = torch.mean(image_t_temp, dim=1).unsqueeze(1).squeeze(0)
    image_s_temp = image_s_temp.squeeze(0)

    image_s_temp = torch.matmul(image_s_temp - means_s, trans_matrix) + means_t
    image_s_temp = torch.clip(image_s_temp, 0, 255)
    image_s_temp = image_s_temp.unsqueeze(0).transpose(2, 1).view(
        input_size_target[0], input_size_target[1],
        input_size_target[2], input_size_target[3]
    )

    return image_s_temp, ass_matrix, weights_out

def image_dmm_lab_pool(dmm, image_s, image_t, label_s, label_t, num_classes, scale=32, wflag=False, pflag=False):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_pool_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_pool(label_s, label_t, num_classes, scale=scale)

    ass_matrix = None
    weights_out = None
    trans_matrix = None
    if wflag:
        ass_ind_1, ass_ind_2, weights, weights_out = assignment_for_arrays_cosine(label_s_vec, F.softmax(label_t_vec, dim=2), wflag=wflag,
                                                     pflag=pflag)
        ass_matrix, weight_matrix = vector2matrix(ass_ind_1, ass_ind_2, weights, length=temp_size[0] * temp_size[1])
        trans_matrix = dmm.get_transferring_matrix(image_s_vec.detach().cpu().numpy(),
                                                   image_t_vec.detach().cpu().numpy(),
                                                   ass_ind_1, ass_ind_2, weights=weights)
        # weight_matrix = torch.tensor(weight_matrix).float()
        weights_out = torch.tensor(weights_out).float().detach()
    else:
        ass_ind_1, ass_ind_2 = assignment_for_arrays_cosine(label_s_vec, F.softmax(label_t_vec, dim=2), wflag=wflag, pflag=pflag)
        ass_matrix = ass_vector2matrix(ass_ind_1, ass_ind_2, length=temp_size[0] * temp_size[1])
        trans_matrix = dmm.get_transferring_matrix(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(),
                                               ass_ind_1, ass_ind_2, None)
    trans_matrix = torch.tensor(trans_matrix).float()

    ngbhd = scale + 1
    pool = nn.AvgPool2d((ngbhd, ngbhd), padding=ngbhd // 2, stride=1)
    image_s_temp = image_s - pool(image_s)
    image_t_mean = pool(image_t)

    image_s_temp = image_s_temp.view(input_size_source[0], input_size_source[1], -1).transpose(2, 1)
    image_t_temp = image_t.view(input_size_target[0], input_size_target[1], -1).transpose(2, 1)
    image_t_mean = image_t_mean.view(input_size_target[0], input_size_target[1], -1).transpose(2, 1)

    # means_s = torch.mean(image_s_temp, dim=1).unsqueeze(1).squeeze(0)
    means_t = torch.mean(image_t_temp, dim=1).unsqueeze(1).squeeze(0)
    image_s_temp = image_s_temp.squeeze(0)

    image_s_temp = torch.matmul(image_s_temp, trans_matrix) + image_t_mean  # + means_t
    # image_s_temp = normalization(image_s_temp, 0, 255)
    image_s_temp = torch.clip(image_s_temp, 0, 255)
    image_s_temp = image_s_temp.unsqueeze(0).transpose(2, 1).view(
        input_size_target[0], input_size_target[1],
        input_size_target[2], input_size_target[3]
    )

    ass_matrix = torch.tensor(ass_matrix).float().detach()
    return image_s_temp, ass_matrix, weights_out



