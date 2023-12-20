#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def image_level_transform(images_s, images_t, sizes=(16, 16)):
    images_s_shape = images_s.shape
    interp_pred = nn.Upsample(size=sizes, mode='bilinear', align_corners=True)

    images_s_temp = interp_pred(images_s)
    images_t_temp = interp_pred(images_t)
    images_s_temp = images_s_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)
    images_t_temp = images_t_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)

    means_s = torch.mean(images_s_temp, dim=1).unsqueeze(1)
    means_t = torch.mean(images_t_temp, dim=1).unsqueeze(1)

    images_s_temp = F.softmax(images_s_temp - means_s, dim=2)
    images_t_temp = F.softmax(images_t_temp - means_t, dim=2)

    # images_s_temp = images_s_temp - means_s
    # images_t_temp = images_t_temp - means_t

    return images_s_temp, images_t_temp

def image_level_pool_transform(images_s, images_t, sizes=(16, 16)):
    images_s_shape = images_s.shape
    interp_pred = nn.Upsample(size=sizes, mode='bilinear', align_corners=True)
    ngbhd = 3
    pool = nn.AvgPool2d((ngbhd, ngbhd), padding=ngbhd//2, stride=1)

    images_s_temp = interp_pred(images_s)
    images_t_temp = interp_pred(images_t)
    images_s_temp = images_s_temp - pool(images_s_temp)
    images_t_temp = images_t_temp - pool(images_t_temp)
    images_s_temp = images_s_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)
    images_t_temp = images_t_temp.view(images_s_shape[0], images_s_shape[1], -1).transpose(2, 1)

    images_s_temp = F.softmax(images_s_temp, dim=2)
    images_t_temp = F.softmax(images_t_temp, dim=2)

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


def cost_matrix_cosine_computation(images_s, images_t):
    shape_image = images_s.shape
    if shape_image[0] != 1:
        return None

    images_s = images_s.squeeze(0)
    images_t = images_t.squeeze(0)
    images_s_n = torch.norm(images_s, p='fro', dim=1).unsqueeze(1) + 1e-12
    images_t_n = torch.norm(images_t, p='fro', dim=1).unsqueeze(1) + 1e-12
    images_s = images_s / images_s_n
    images_t = images_t / images_t_n

    cost_matrix = torch.mm(images_s, torch.transpose(images_t, 1, 0))
    cost_matrix = torch.exp(-cost_matrix)
    return cost_matrix

def cost_matrix_l2_computation(images_s, images_t, p=2):
    shape_image = images_s.shape
    if shape_image[0] != 1:
        return None

    x_col = images_s.unsqueeze(-2)
    y_lin = images_t.unsqueeze(-3)
    cost_matrix = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    cost_matrix = cost_matrix.squeeze(0)

    return cost_matrix

def cost_matrix_cosine(A, B, dim=-1, eps=1e-8):
    numerator = torch.bmm(A, torch.permute(B, (0, 2, 1)))
    A_l2 = torch.mul(A, A).sum(axis=dim).unsqueeze(-1)
    B_l2 = torch.mul(B, B).sum(axis=dim).unsqueeze(-1)
    denominator = torch.max(torch.sqrt(torch.bmm(A_l2, torch.permute(B_l2, (0, 2, 1)))), torch.tensor(eps))
    weight_matrix = torch.div(numerator, denominator).squeeze(0)
    cost_matrix = weight_matrix.max() - weight_matrix + eps
    return cost_matrix, weight_matrix


def assigned_matrix_for_arrays(array_data1, array_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    cost_matrix = cost_matrix_l2_computation(array_data1, array_data2)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        pi[r, c] = 1
    if device == 'cuda':
        pi = torch.tensor(pi).cuda()

    return pi


def assigned_matrix_computation(image_data1, image_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        pi[r, c] = 1
    if device == 'cuda':
        pi = torch.tensor(pi).cuda()

    return pi


def assigned_matrix_computation_thred(image_data1, image_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    thred = np.median(cost_matrix[row_ind, col_ind])
    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < thred:
            pi[r, c] = 1
    if device == 'cuda':
        pi = torch.tensor(pi).cuda()

    return pi


def assigned_matrix_weights_thred(image_data1, image_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp)
    cost_matrix = cost_matrix.cpu().numpy()
    weights = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    thred = np.median(cost_matrix[row_ind, col_ind])
    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < thred:
            pi[r, c] = 1
            weights[r, c] = np.exp(-cost_matrix[r, c])
    weights = normalization(weights, 0, 1, eps=1e-4)
    if device == 'cuda':
        pi = torch.tensor(pi).cuda()
        weights = torch.tensor(weights).cuda()

    return pi, weights

def assigned_matrix_weights(image_data1, image_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp, p=1)
    cost_matrix = cost_matrix.cpu().numpy()
    weights = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        pi[r, c] = 1
        weights[r, c] = np.exp(-2.0*cost_matrix[r, c])
    weights = normalization_nonzeros(weights, 0, 1, eps=1e-4)
    if device == 'cuda':
        pi = torch.tensor(pi).cuda()
        weights = torch.tensor(weights).cuda()

    return pi, weights

def assigned_matrix_cost(image_data1, image_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp, p=1)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        pi[r, c] = 1

    if device == 'cuda':
        pi = torch.tensor(pi).cuda()
        cost_matrix = torch.tensor(cost_matrix).cuda()

    return pi, cost_matrix

def normalization(x, min, max, eps=1e-6):
    x_min = x.min()
    x_max = x.max()
    x_nor = (x - x_min) * (max - min) / (x_max - x_min + eps) + min
    return x_nor

def normalization_nonzeros(x, min, max, eps=1e-6):
    x_min = np.min(x[x>eps])
    x_max = np.max(x)
    x_nor = (x - x_min) * (max - min) / (x_max - x_min + eps) + min
    x_nor = np.clip(x_nor, min + eps, max - eps)
    return x_nor

def attention_map(image_data1, image_data2):
    image_data1_temp = image_data1.max(dim=1)[0].unsqueeze(1)
    image_data2_temp = image_data2.max(dim=1)[0].unsqueeze(1)
    shape = image_data2_temp.shape

    image_data1_temp = image_data1_temp.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2_temp.view(shape[0], shape[1], -1)

    attention = torch.matmul(image_data1_temp, image_data2_temp)
    max_v = attention.max() + 1e-12
    attention = attention / max_v
    return attention


def assigned_loss_computation(image_data1, image_data2, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp)
    cost_matrix = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        pi[r, c] = 1
    if device == 'cuda':
        pi = torch.tensor(pi).cuda()
        cost_matrix = torch.tensor(cost_matrix).cuda()

    pi = pi.unsqueeze(0)
    cost_matrix = cost_matrix.unsqueeze(0)
    cost = torch.sum(pi * cost_matrix, dim=(-2, -1)) / pi.shape[-1]
    # cost = torch.sum(pi * cost_matrix, dim=(-2, -1))
    cost = cost.mean()

    return cost

def weighted_assigned_matrix_for_arrays(array_data1, array_data2):
    from scipy.optimize import linear_sum_assignment
    cost_matrix = cost_matrix_l2_computation(array_data1, array_data2)
    # cost_matrix = cost_matrix_l2_computation_scipy(array_data1, array_data2)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    weights = np.zeros(cost_matrix.shape[-2])
    pi = np.zeros([cost_matrix.shape[-2], cost_matrix.shape[-1]])
    for r, c in zip(row_ind, col_ind):
        pi[r, c] = 1
        weights[r] = np.exp(-cost_matrix[r, c])

    return col_ind, pi, weights

def weighted_assigned_vector_for_arrays(array_data1, array_data2):
    from scipy.optimize import linear_sum_assignment
    cost_matrix = cost_matrix_l2_computation(array_data1, array_data2)
    # cost_matrix = cost_matrix_l2_computation_scipy(array_data1, array_data2)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    Thred = np.median(cost_matrix[row_ind, col_ind])
    weights = []
    ass_ind_1 = []
    ass_ind_2 = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < Thred:
            ass_ind_1.append(r)
            ass_ind_2.append(c)
            weights.append(np.exp(-cost_matrix[r, c]))

    return ass_ind_1, ass_ind_2, weights

def sweighted_assigned_vector_for_arrays(array_data1, array_data2):
    from scipy.optimize import linear_sum_assignment
    cost_matrix = cost_matrix_l2_computation(array_data1, array_data2, p=1)
    # cost_matrix = cost_matrix_l2_computation_scipy(array_data1, array_data2)
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    cost_matrix = np.exp(-2.0*cost_matrix)
    weights = []
    ass_ind_1 = []
    ass_ind_2 = []
    for r, c in zip(row_ind, col_ind):
        ass_ind_1.append(r)
        ass_ind_2.append(c)
        weights.append(cost_matrix[r, c])

    # weights = normalization(weights, 0, 1, eps=1e-6)
    return ass_ind_1, ass_ind_2, weights

def assignment_for_arrays(array_data1, array_data2, wflag=False, pflag=False):
    from scipy.optimize import linear_sum_assignment
    # *** The l2 cost is also good, you can try it. ***#
    # cost_matrix = cost_matrix_l2_computation(array_data1, array_data2, p=2)
    # weight_matrix = []
    # cost_matrix = cost_matrix.cpu()
    # if wflag:
    #     weight_matrix = np.exp(-4.0*cost_matrix)

    cost_matrix, weight_matrix = cost_matrix_cosine(array_data1, array_data2, dim=-1)

    weights = []
    ass_ind_1 = []
    ass_ind_2 = []

    gn = cost_matrix.shape[0] + 1 # cost_matrix.max()*cost_matrix.shape[0] + 1 # 500
    cost_matrix = cost_greatnum_from_preds(cost_matrix, array_data1, array_data2, gn=gn)
    if cost_matrix[cost_matrix != gn].shape[0] == 0:
        ass_ind_1 = list(range(0, cost_matrix.shape[-2]))
        ass_ind_2 = list(range(0, cost_matrix.shape[-1]))
        if wflag:
            weights = np.ones(cost_matrix.shape[-2]).tolist()
            return ass_ind_1, ass_ind_2, weights, weight_matrix
        else:
            return ass_ind_1, ass_ind_2
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pvalue = gn
    if pflag:
        percent = row_ind.shape[0] / cost_matrix[cost_matrix != gn].shape[0] * 100.0
        if percent > 0 and percent < 100:
            pvalue = np.percentile(cost_matrix[cost_matrix != gn], percent)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < pvalue:
            ass_ind_1.append(r)
            ass_ind_2.append(c)
            if wflag:
                weights.append(weight_matrix[r, c])

    if len(ass_ind_1) == 0:
        ass_ind_1 = list(range(0, cost_matrix.shape[-2]))
        ass_ind_2 = list(range(0, cost_matrix.shape[-1]))
        if wflag:
            weights = np.ones(cost_matrix.shape[-2]).tolist()

    if wflag:
        return ass_ind_1, ass_ind_2, weights, weight_matrix
    else:
        return ass_ind_1, ass_ind_2

def assignment_for_arrays_cosine(array_data1, array_data2, wflag=False, pflag=False):
    from scipy.optimize import linear_sum_assignment
    cost_matrix, weight_matrix = cost_matrix_cosine(array_data1, array_data2, dim=-1)
    weight_matrix = normalization(weight_matrix, 0.0, 1.0)
    cost_matrix = cost_matrix.cpu()

    weights = []
    ass_ind_1 = []
    ass_ind_2 = []

    gn = cost_matrix.shape[0] + 1 # 500
    cost_matrix = cost_greatnum_from_preds(cost_matrix, array_data1, array_data2, gn=gn)
    if cost_matrix[cost_matrix != gn].shape[0] == 0:
        ass_ind_1 = list(range(0, cost_matrix.shape[-2]))
        ass_ind_2 = list(range(0, cost_matrix.shape[-1]))
        if wflag:
            weights = np.ones(cost_matrix.shape[-2]).tolist()
            return ass_ind_1, ass_ind_2, weights, weight_matrix
        else:
            return ass_ind_1, ass_ind_2
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pvalue = gn
    if pflag:
        percent = row_ind.shape[0] / cost_matrix[cost_matrix != gn].shape[0] * 100.0
        if percent > 0 and percent < 100:
            pvalue = np.percentile(cost_matrix[cost_matrix != gn], percent)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < pvalue:
            ass_ind_1.append(r)
            ass_ind_2.append(c)
            if wflag:
                weights.append(weight_matrix[r, c])

    if len(ass_ind_1) == 0:
        ass_ind_1 = list(range(0, cost_matrix.shape[-2]))
        ass_ind_2 = list(range(0, cost_matrix.shape[-1]))
        if wflag:
            weights = np.ones(cost_matrix.shape[-2]).tolist()

    if wflag:
        return ass_ind_1, ass_ind_2, weights, weight_matrix
    else:
        return ass_ind_1, ass_ind_2



def cost_greatnum_from_preds(cost_matrix, pred1, pred2, gn=500):
    cost_matrix.to(pred1.device)
    matrix_size = [cost_matrix.shape[-2], cost_matrix.shape[-1]]
    lab1 = pred1.max(dim=2)[1].squeeze(0)
    lab2 = pred2.max(dim=2)[1].squeeze(0)
    red_count = redundancy_detection(lab1, lab2)

    for r in range(matrix_size[0]):
        cur_r_lab = lab1[r]
        cost_matrix[r, :][lab2 != cur_r_lab] = gn

    return cost_matrix


def reverse_normalization_exp(x, min, max, eps=1e-6):
    x_nor = np.exp(-x)
    x_min = np.min(x_nor)
    x_max = np.max(x_nor)
    x_nor = (x_nor - x_min) * (max - min) / (x_max - x_min + eps) + min
    x_nor = np.clip(x_nor, min + eps, max - eps)
    return x_nor

def redundancy_detection(lab1, lab2):
    labs = torch.unique(lab1)
    count = 0

    for l in labs:
        count_1 = lab1[torch.eq(lab1, l)].shape[0]
        count_2 = lab2[torch.eq(lab2, l)].shape[0]
        if count_1 > count_2:
            count = count + count_1 - count_2

    return count



