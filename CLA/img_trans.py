#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import numpy as np
import torch
import torch.nn as nn
from DA.CLA.opt_tools import *
from DA.CLA.DMM import DMM


def image_level_transform(images_s, images_t, sizes=(16, 16)):
    images_s_shape = images_s.shape
    images_t_shape = images_t.shape
    interp_pred = nn.Upsample(size=sizes, mode='bilinear', align_corners=True)

    images_s_temp = interp_pred(images_s)
    images_t_temp = interp_pred(images_t)
    images_s_temp = images_s_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)
    images_t_temp = images_t_temp.view(images_t_shape[0], images_t_shape[1], -1).transpose(2, 1)

    means_s = torch.mean(images_s_temp, dim=1).unsqueeze(1)
    means_t = torch.mean(images_t_temp, dim=1).unsqueeze(1)

    images_s_temp = F.softmax(images_s_temp - means_s, dim=2)
    images_t_temp = F.softmax(images_t_temp - means_t, dim=2)

    # images_s_temp = images_s_temp - means_s
    # images_t_temp = images_t_temp - means_t

    return images_s_temp, images_t_temp

def image_level_transform_predown(images_s, images_t, interp_down):
    images_s_shape = images_s.shape
    images_t_shape = images_t.shape

    images_s_temp = interp_down(images_s)
    images_t_temp = interp_down(images_t)
    images_s_temp = images_s_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)
    images_t_temp = images_t_temp.view(images_t_shape[0], images_t_shape[1], -1).transpose(2, 1)

    means_s = torch.mean(images_s_temp, dim=1).unsqueeze(1)
    means_t = torch.mean(images_t_temp, dim=1).unsqueeze(1)

    images_s_temp = F.softmax(images_s_temp - means_s, dim=2)
    images_t_temp = F.softmax(images_t_temp - means_t, dim=2)

    # images_s_temp = images_s_temp - means_s
    # images_t_temp = images_t_temp - means_t

    return images_s_temp, images_t_temp

def image_level_transform_without_downs(images_s, images_t):
    images_s_shape = images_s.shape

    images_s_temp = images_s
    images_t_temp = images_t
    images_s_temp = images_s_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)
    images_t_temp = images_t_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)

    means_s = torch.mean(images_s_temp, dim=1).unsqueeze(1)
    means_t = torch.mean(images_t_temp, dim=1).unsqueeze(1)

    images_s_temp = F.softmax(images_s_temp - means_s, dim=2)
    images_t_temp = F.softmax(images_t_temp - means_t, dim=2)

    return images_s_temp, images_t_temp


def image_label_alignment(image_s_vec, image_t_vec, label_s, label_t, num_classes):
    image_shape = image_s_vec.shape
    image_s_h = int(image_s_vec.shape[1] ** 0.5)
    interp_pred = nn.Upsample(size=(image_s_h, image_s_h), mode='nearest')

    out_s = interp_pred(label_s.unsqueeze(1)).squeeze(0)
    out_s = label_2_onehot(out_s, num_classes)
    out_t = interp_pred(label_t)
    out_label_s = out_s.view(image_shape[0], num_classes, -1).transpose(2, 1)
    out_label_t = out_t.view(image_shape[0], num_classes, -1).transpose(2, 1)

    out_s = torch.cat([image_s_vec, out_label_s], dim=2)
    out_t = torch.cat([image_t_vec, out_label_t], dim=2)

    return out_s, out_t, out_label_s, out_label_t


def label_2_onehot(label, num_classes):
    onehot_out = F.one_hot(label.data[0].long(), num_classes)
    onehot_out = onehot_out.unsqueeze(0).permute(0, 3, 1, 2)
    return onehot_out


def labels_transform(label_s, label_t, num_classes, sizes=(16, 16)):
    image_shape = label_t.shape
    # interp_pred = nn.Upsample(size=sizes, mode='nearest')
    interp_pred = nn.Upsample(size=sizes, mode='bilinear')

    out_s = interp_pred(label_s.unsqueeze(1)).squeeze(0)
    out_s = label_2_onehot(out_s, num_classes)
    out_t = interp_pred(label_t)
    out_label_s = out_s.view(image_shape[0], num_classes, -1).transpose(2, 1)
    out_label_t = out_t.view(image_shape[0], num_classes, -1).transpose(2, 1)

    return out_label_s, out_label_t


def labels_transform_aug(label_s, label_t, num_classes, sizes=(16, 16)):
    image_shape = label_t.shape
    # interp_pred = nn.Upsample(size=sizes, mode='nearest')
    interp_pred = nn.Upsample(size=sizes, mode='bilinear')

    label_s_temp = label_s.clone()
    label_s_temp[label_s >= num_classes] = num_classes
    label_s_temp[label_s < 0] = num_classes
    out_s = interp_pred(label_s_temp.unsqueeze(1)).squeeze(0)
    out_s = label_2_onehot(out_s, num_classes+1)
    zero_pad = torch.zeros(1, 1, label_t.shape[-2], label_t.shape[-1]).to(label_t.device)
    label_t_temp = torch.cat((label_t, zero_pad), dim=1)
    out_t = interp_pred(label_t_temp)
    out_label_s = out_s.view(image_shape[0], num_classes+1, -1).transpose(2, 1)
    out_label_t = out_t.view(image_shape[0], num_classes+1, -1).transpose(2, 1)

    return out_label_s, out_label_t

def labels_transform_pool_aug(label_s, label_t, num_classes, scale=32):
    image_shape = label_s.shape
    interp_pred = nn.Upsample(size=(image_shape[-2]//scale, image_shape[-1]//scale), mode='bilinear')
    down_raw = nn.AvgPool2d((scale*2, scale*2), padding=scale - 1, stride=scale)

    label_s_temp = label_s.clone()
    label_s_temp[label_s >= num_classes] = num_classes
    label_s_temp[label_s < 0] = num_classes
    out_s = label_2_onehot(label_s_temp, num_classes + 1).float()
    out_s = down_raw(out_s)

    zero_pad = torch.zeros(1, 1, label_t.shape[-2], label_t.shape[-1]).to(label_t.device)
    label_t_temp = torch.cat((label_t, zero_pad), dim=1)
    out_t = interp_pred(label_t_temp)
    out_label_s = out_s.view(image_shape[0], num_classes+1, -1).transpose(2, 1)
    out_label_t = out_t.view(image_shape[0], num_classes+1, -1).transpose(2, 1)

    return out_label_s, out_label_t



def labels_transform_aug_predown(label_s, label_t, num_classes, down_raw, down_pred):
    image_shape = label_t.shape

    label_s_temp = label_s.clone()
    label_s_temp[label_s >= num_classes] = num_classes
    label_s_temp[label_s < 0] = num_classes
    out_s = down_raw(label_s_temp.unsqueeze(1)).squeeze(0)
    out_s = label_2_onehot(out_s, num_classes+1)
    zero_pad = torch.zeros(1, 1, label_t.shape[-2], label_t.shape[-1]).to(label_t.device)
    label_t_temp = torch.cat((label_t, zero_pad), dim=1)
    out_t = down_pred(label_t_temp)
    out_label_s = out_s.view(image_shape[0], num_classes+1, -1).transpose(2, 1)
    out_label_t = out_t.view(image_shape[0], num_classes+1, -1).transpose(2, 1)

    return out_label_s, out_label_t


def image_trans_l2_withLab(image_s, image_t, label_s, label_t, num_classes):
    input_size_source = image_s.shape
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=(
        input_size_source[-2] // 32, input_size_source[-1] // 32))
    image_s_temp, image_t_temp, label_s_vec, label_t_vec = image_label_alignment(image_s_vec, image_t_vec, label_s, label_t, num_classes)
    pi_l = ott.assigned_matrix_for_arrays(label_s_vec, label_t_vec)

    pi, pi_m = assigned_matrix_for_arrays(image_s_temp, image_t_temp)
    trans_matrix = optimization_l2(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), pi)
    trans_matrix = torch.tensor(trans_matrix).float()

    image_s_temp = image_s.view(input_size_source[0], input_size_source[1], -1).transpose(2, 1)
    image_t_temp = image_t.view(input_size_source[0], input_size_source[1], -1).transpose(2, 1)

    means_s = torch.mean(image_s_temp, dim=1).unsqueeze(1).squeeze(0)
    means_t = torch.mean(image_t_temp, dim=1).unsqueeze(1).squeeze(0)
    image_s_temp = image_s_temp.squeeze(0)

    image_s_temp = torch.matmul(image_s_temp - means_s, trans_matrix) + means_t
    image_s_temp = torch.clip(image_s_temp, 0, 255)
    image_s_temp = image_s_temp.unsqueeze(0).transpose(2, 1).view(
        input_size_source[0], input_size_source[1],
        input_size_source[2], input_size_source[3]
    )

    return image_s_temp, pi_l


def image_trans_l2_Lab(image_s, image_t, label_s, label_t, num_classes, scale=32):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_aug(label_s, label_t, num_classes, sizes=temp_size)
    # pi_l = ott.assigned_matrix_for_arrays(label_s_vec, label_t_vec)

    pi, pi_m = assigned_matrix_for_arrays(label_s_vec, F.softmax(label_t_vec, dim=2))
    trans_matrix = optimization_l2(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), pi)
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

    return image_s_temp, pi_m

def image_trans_l2_Lab_predown(image_s, image_t, label_s, label_t, num_classes, down_raw, down_pred):
    input_size_source = image_s.shape
    input_size_target = image_t.shape

    image_s_vec, image_t_vec = image_level_transform_predown(image_s, image_t, down_raw)
    label_s_vec, label_t_vec = labels_transform_aug_predown(label_s, label_t, num_classes, down_raw, down_pred)

    pi, pi_m = assigned_matrix_for_arrays(label_s_vec, F.softmax(label_t_vec, dim=2))
    trans_matrix = optimization_l2(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), pi)
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

    return image_s_temp, pi_m


def image_trans_l2_Lab_weights(image_s, image_t, label_s, label_t, num_classes, scale=16):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_aug(label_s, label_t, num_classes, sizes=temp_size)
    # pi_l = ott.assigned_matrix_for_arrays(label_s_vec, label_t_vec)

    pi, pi_m, weights = weighted_assigned_matrix_for_arrays(label_s_vec, F.softmax(label_t_vec, dim=2))
    trans_matrix = weighted_optimization_l2(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), pi, weights)
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

    return image_s_temp, pi_m

def image_trans_l2_Lab_unb_weights(image_s, image_t, label_s, label_t, num_classes, scale=32):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_aug(label_s, label_t, num_classes, sizes=temp_size)
    # pi_l = ott.assigned_matrix_for_arrays(label_s_vec, label_t_vec)

    ass_ind_1, ass_ind_2, weights = sweighted_assigned_vector_for_arrays(label_s_vec, F.softmax(label_t_vec, dim=2))
    trans_matrix = weighted_unb_optimization_l2(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), ass_ind_1, ass_ind_2, weights)
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

    return image_s_temp


def image_dmm_l2_Lab_unb_weights(dmm, image_s, image_t, label_s, label_t, num_classes, scale=32):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_pool_aug(label_s, label_t, num_classes, scale=scale)
    # pi_l = ott.assigned_matrix_for_arrays(label_s_vec, label_t_vec)

    ass_ind_1, ass_ind_2, weights = sweighted_assigned_vector_for_arrays(label_s_vec, F.softmax(label_t_vec, dim=2))
    # trans_matrix = weighted_unb_optimization_l2(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), ass_ind_1, ass_ind_2, weights)
    trans_matrix = dmm.get_transferring_matrix(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), ass_ind_1, ass_ind_2, weights)
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

    return image_s_temp

def image_dmm_l2_Lab_unb_weights_out(dmm, image_s, image_t, label_s, label_t, num_classes, scale=32):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)
    image_s_vec, image_t_vec = image_level_transform(image_s, image_t, sizes=temp_size)
    label_s_vec, label_t_vec = labels_transform_pool_aug(label_s, label_t, num_classes, scale=scale)
    # pi_l = ott.assigned_matrix_for_arrays(label_s_vec, label_t_vec)

    ass_ind_1, ass_ind_2, weights = sweighted_assigned_vector_for_arrays(label_s_vec, F.softmax(label_t_vec, dim=2))
    ass_matrix, weight_matrix = vector2matrix(ass_ind_1, ass_ind_2, weights)
    trans_matrix = dmm.get_transferring_matrix(image_s_vec.detach().cpu().numpy(), image_t_vec.detach().cpu().numpy(), ass_ind_1, ass_ind_2, weights)
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

    ass_matrix = torch.tensor(ass_matrix).detach()
    weight_matrix = torch.tensor(weight_matrix).detach()
    return image_s_temp, ass_matrix, weight_matrix

def vector2matrix(ass_ind_1, ass_ind_2, weights):
    ass_matrix = np.zeros([len(ass_ind_1), len(ass_ind_2)])
    weight_matrix = np.zeros([len(ass_ind_1), len(ass_ind_2)])
    count = 0
    for r, c in zip(ass_ind_1, ass_ind_2):
        ass_matrix[r, c] = 1.0
        weight_matrix[r, c] = weights[count]
        count = count + 1

    return ass_matrix, weight_matrix

def image_dmm_Lab_unb_weights(dmm, image_s, image_t, label_s, label_t, num_classes, scale=32):
    input_size_source = image_s.shape
    input_size_target = image_t.shape
    temp_size = (input_size_source[-2] // scale, input_size_source[-1] // scale)

    image_s_v = image_s.view(input_size_source[0], input_size_source[1], -1).transpose(2, 1)
    image_t_v = image_t.view(input_size_target[0], input_size_target[1], -1).transpose(2, 1)
    means_s = torch.mean(image_s_v, dim=1).unsqueeze(1).squeeze(0)
    means_t = torch.mean(image_t_v, dim=1).unsqueeze(1).squeeze(0)
    means_s = means_s.unsqueeze(-1).unsqueeze(-1)
    means_t = means_t.unsqueeze(-1).unsqueeze(-1)

    vecs = data_reduction(image_s-means_s, image_t-means_t, label_s, label_t, num_classes, scale=scale)
    ass_ind_1, ass_ind_2, weights = sweighted_assigned_vector_for_arrays(vecs[-2], F.softmax(vecs[-1], dim=2))

    # ass_matrix, weight_matrix = vector2matrix(ass_ind_1, ass_ind_2, weights)
    trans_matrix = dmm.get_transferring_matrix(vecs[0].detach().cpu().numpy(), vecs[1].detach().cpu().numpy(), ass_ind_1, ass_ind_2, weights)
    trans_matrix = torch.tensor(trans_matrix).float()

    means_s = means_s.squeeze(-1).squeeze(-1)
    means_t = means_t.squeeze(-1).squeeze(-1)
    image_s_v = torch.matmul(image_s_v - means_s, trans_matrix) + means_t
    image_s_v = torch.clip(image_s_v, 0, 255)
    image_s_v = image_s_v.transpose(2, 1).view(
        input_size_target[0], input_size_target[1],
        input_size_target[2], input_size_target[3]
    )

    # ass_matrix = torch.tensor(ass_matrix).detach()
    # weight_matrix = torch.tensor(weight_matrix).detach()
    # return image_s_temp, ass_matrix, weight_matrix
    return image_s_v

def data_reduction(image_s, image_t, label_s, label_t, num_classes, scale=16):
    image_s_shape = image_s.shape
    image_t_shape = image_t.shape
    interp_image = nn.Upsample(size=[image_s_shape[-2]//scale, image_s_shape[-1]//scale], mode='bilinear', align_corners=True)
    interp_label = nn.AvgPool2d((scale * 2, scale * 2), padding=scale - 1, stride=scale)

    image_s_temp = F.softmax(interp_image(image_s), dim=1)
    image_t_temp = F.softmax(interp_image(image_t), dim=1)
    # image_s_temp = interp_image(image_s)
    # image_t_temp = interp_image(image_t)

    label_s_temp = label_s.clone()
    label_s_temp[label_s >= num_classes] = num_classes
    label_s_temp[label_s < 0] = num_classes
    label_s_temp = label_2_onehot(label_s_temp, num_classes).float()
    label_s_temp = interp_label(label_s_temp)
    label_t_temp = interp_image(label_t)

    label_s_map = torch.max(label_s_temp, dim=1)[1]
    label_t_map = torch.max(label_t_temp, dim=1)[1]
    label_s_set = torch.unique(label_s_map)
    label_t_set = torch.unique(label_t_map)

    inter_sec, diff = inter_diff(label_s_set, label_t_set)

    if diff.shape[0] == 0 or inter_sec.shape[0] == 0:
        image_s_temp = image_s_temp.view(image_s_shape[0], image_s_shape[1], -1).transpose(2, 1)
        image_t_temp = image_t_temp.view(image_t_shape[0], image_t_shape[1], -1).transpose(2, 1)
        label_s_temp = label_s_temp.view(image_s_shape[0], num_classes, -1).transpose(2, 1)
        label_t_temp = label_t_temp.view(image_s_shape[0], num_classes, -1).transpose(2, 1)
    else:
        flag_s = torch.ones([label_t_map.shape[-2], label_t_map.shape[-1]], dtype=torch.bool)
        flag_t = torch.ones([label_t_map.shape[-2], label_t_map.shape[-1]], dtype=torch.bool)
        for i in diff:
            flag_s[label_s_map.squeeze(0) == i.item()] = False
            flag_t[label_t_map.squeeze(0) == i.item()] = False
        image_s_temp = image_s_temp[:, :, flag_s].transpose(2, 1)
        image_t_temp = image_t_temp[:, :, flag_t].transpose(2, 1)
        label_s_temp = label_s_temp[:, :, flag_s].transpose(2, 1)
        label_t_temp = label_t_temp[:, :, flag_t].transpose(2, 1)

    return image_s_temp, image_t_temp, label_s_temp, label_t_temp

def inter_diff(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return intersection, difference



