#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

mce_loss = nn.MSELoss()


def channel_1toN(img, num_channel):
    T = torch.LongTensor(num_channel, img.shape[1], img.shape[2]).zero_()
    mask = torch.LongTensor(img.shape[1], img.shape[2]).zero_()
    for i in range(num_channel):
        T[i] = T[i] + i
        layer = T[i] - img
        T[i] = torch.from_numpy(np.logical_not(np.logical_xor(layer.numpy(), mask.numpy())).astype(int))
    return T.float()


class WeightedBCEWithLogitsLoss(nn.Module):
    
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        
    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
                
        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


class CrossEntropy2d(nn.Module):
    
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax2d()
        
        P = sm(predict)
        P = torch.clamp(P, min = 1e-9, max = 1-(1e-9))
        
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].view(1, -1)
        predict = P[target_mask.view(N, 1, H, W).repeat(1, C, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim = 0, index = target)
        log_p = probs.log()
        batch_loss = -(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            
            loss = batch_loss.sum()
        return loss


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, device="cpu"):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class SinkhornDistanceGivenC(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistanceGivenC, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, C, device="cpu"):
        # The Sinkhorn algorithm takes as input three variables :
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class OTDistanceGivenPI(nn.Module):

    def __init__(self, reduction='mean'):
        super(OTDistanceGivenPI, self).__init__()
        self.reduction = reduction

    def forward(self, x, y, pi, device="cpu"):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function

        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, C

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C


class CostGivenAssignmentsAndWeights(nn.Module):

    def __init__(self, p=2, reduction='mean'):
        super(CostGivenAssignmentsAndWeights, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, x, y, pi, weights):
        C = self._cost_matrix(x, y, p=self.p)
        # C = self._cost_matrix_cosine(x, y, dim=-1)

        cost = torch.sum(pi * C * weights, dim=(-2, -1)) / pi.shape[-1]
        # cost = torch.sum(pi * C * weights, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, C

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    def _cost_matrix_cosine(self, A, B, dim=-1, eps=1e-8):
        numerator = torch.bmm(A, torch.permute(B, (0, 2, 1)))
        # A_l2 = torch.mul(A, A).sum(axis=dim)
        # B_l2 = torch.mul(B, B).sum(axis=dim)
        # denominator = torch.max(torch.sqrt(torch.mul(A_l2, B_l2)), torch.tensor(eps))
        # C = torch.div(numerator, denominator.unsqueeze(-1))

        A_l2 = torch.mul(A, A).sum(axis=dim).unsqueeze(-1)
        B_l2 = torch.mul(B, B).sum(axis=dim).unsqueeze(-1)
        denominator = torch.max(torch.sqrt(torch.bmm(A_l2, torch.permute(B_l2, (0, 2, 1)))), torch.tensor(eps))
        C = torch.div(numerator, denominator)
        # print('max: ', C.max())
        C = torch.tensor(1.0) - C + eps
        return C

class CostGivenAssignments(nn.Module):

    def __init__(self, p=2, reduction='mean'):
        super(CostGivenAssignments, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, x, y, pi):
        C = self._cost_matrix(x, y, p=self.p)

        cost = torch.sum(pi * C, dim=(-2, -1)) / pi.shape[-1]
        # cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, C

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

class CostMultiplication(nn.Module):
    def __init__(self, reduction='mean'):
        super(CostMultiplication, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2):
        cost = torch.sum(x1 * x2, dim=(-2, -1)) / x1.shape[-1]
        # cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

def cost_matrix_l2_computation(images_s, images_t, p=2):
    shape_image = images_s.shape
    if shape_image[0] != 1:
        return None

    x_col = images_s.unsqueeze(-2)
    y_lin = images_t.unsqueeze(-3)
    cost_matrix = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    cost_matrix = cost_matrix.squeeze(0)

    return cost_matrix

def cost_matrix(image_data1, image_data2):
    shape = image_data1.shape
    image_data1_temp = image_data1.view(shape[0], shape[1], -1).transpose(2, 1)
    image_data2_temp = image_data2.view(shape[0], shape[1], -1).transpose(2, 1)
    cost_matrix = cost_matrix_l2_computation(image_data1_temp, image_data2_temp)
    return cost_matrix

def assignment(cost_matrix, device='cuda'):
    from scipy.optimize import linear_sum_assignment
    cost_matrix = cost_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    if device == 'cuda':
        row_ind = torch.tensor(row_ind).cuda()
        col_ind = torch.tensor(col_ind).cuda()

    return row_ind, col_ind

def assignmentcost(row_ind, col_ind, cost_matrix):
    cost = torch.tensor(0.0)
    thred = torch.median(cost_matrix[row_ind, col_ind])
    count = 0.0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < thred:
            cost = cost + cost_matrix[r, c]
            count = count + 1.0
    cost = cost / count
    return cost



