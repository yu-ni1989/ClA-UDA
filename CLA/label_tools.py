import argparse
from torch.autograd import Variable
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os.path as osp
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F


def num(i, j, size, ol):
    if i < size or j < size:
        print('错误！图像太小了，远小于模型预测所需尺寸！')
    i_stop = int((i - size) / (size - ol)) + 2
    j_stop = int((j - size) / (size - ol)) + 2
    return i_stop, j_stop


def pseudo_label(predT):
    # 伪标签预测
    output = F.softmax(torch.tensor(predT), dim=2)
    output = output.data.numpy()
    # output = output.transpose(1, 2, 0)

    labelT, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # 返回索引，返回数值
    predicted_label = labelT.copy()  # 预测标签
    predicted_prob = prob.copy()  # 预测概率

    thres = []  # 阈值
    for i in range(6):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[int(np.round(len(x) * 0.5))])  # 取中间值
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    for i in range(6):
        labelT[(prob < thres[i]) * (labelT == i)] = 255  # 提取高置信度像素赋予伪标签
    return labelT





