#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp

from DA.CLA.generator import Res_Deeplab
from DA.CLA.loss import CrossEntropy2d
from DA.CLA.datasets_da import PotsdamDatasetWithScale, UAVidTrainWithScale
from DA.CLA.loss import CostGivenAssignmentsAndWeights, CostGivenAssignments
from DA.CLA.DMM import DMM
from DA.CLA.val_tools import uavidtesting
from DA.CLA.opt_tools import assignment_for_arrays
import cv2
from DA.CLA.trans_tools import image_dmm_lab_nonst, ass_vector2matrix, vector2matrix

IMG_MEAN = np.array((0, 0, 0), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 6

MOMENTUM = 0.9
NUM_CLASSES = 6
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/ClA_p-uavid_704-wot/'

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4  # 2.5e-4
LEARNING_RATE_D = 5e-5  #
NUM_STEPS = 50000
NUM_STEPS_STOP = 50000  # Use damping instead of early stopping
PREHEAT_STEPS = 0
POWER = 0.9
RANDOM_SEED = 1234

SOURCE = 'potsdamIRRG'
TARGET = 'UAVID'

INPUT_SIZE = '704,704'
DATA_DIRECTORY = 'D:/NIHUAN/Data/ISPRS_DATA/potsdam/IRRG'
DATA_LIST_PATH = 'D:/NIHUAN/Data/ISPRS_DATA/potsdam/IRRG/images_train.txt'
Lambda_weight = 0.01
Lambda_adv = 0.001
Lambda_local = 40
Epsilon = 0.4

DATA_DIRECTORY_TARGET = 'D:/NIHUAN/Data/UAVid/train_pots'
DATA_DIRECTORY_TEST_TARGET = 'D:/NIHUAN/Data/UAVid/valid_pots'
SCALE = 32


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-dir-test-target", type=str, default=DATA_DIRECTORY_TEST_TARGET)
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--mean", type=float, default=IMG_MEAN)
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)

def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
             (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output

def weightmap_i(pred1, pred2):
    output = torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
             (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output

def main():
    """Create the model and start the training."""
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    model = Res_Deeplab(num_classes=args.num_classes)
    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not args.num_classes == NUM_CLASSES or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    if args.restore_from[:4] == './mo':
        model.load_state_dict(new_params)
    else:
        model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    seg_criterion = CrossEntropy2d(NUM_CLASSES, ignore_label=args.ignore_label).cuda()
    assignc_w = CostGivenAssignmentsAndWeights(p=2).cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        PotsdamDatasetWithScale(args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                                crop_size=input_size, mean=IMG_MEAN, scale=[0.5, 10], mirr=True,
                                ignore_label=args.ignore_label, mark='IRRG', subfold='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(
        UAVidTrainWithScale(args.data_dir_target, max_iters=args.num_steps * args.iter_size * args.batch_size,
                              crop_size=input_size, mean=IMG_MEAN, scale=None, mirr=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp_source = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    interp_pred = nn.Upsample(size=(input_size[1] // SCALE, input_size[0] // SCALE), mode='bilinear',
                              align_corners=True)

    dmm = DMM(3, 3)
    max_miou = 0.0
    max_iter = 0

    for i_iter in range(args.num_steps):
        model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _ = batch
        _, batch = next(targetloader_iter)
        images_t, _, _, _ = batch

        images_t = Variable(images_t).cuda(args.gpu)
        gfeats_t, pred_target = model(images_t)
        pred_target_temp = F.softmax(pred_target, dim=1)

        dmm.reset()
        with torch.no_grad():
            images_s, ass_matrix, weight_matrix = image_dmm_lab_nonst(dmm, images_s, images_t.detach().cpu(), labels_s,
                                                                      pred_target_temp.detach().cpu(), args.num_classes,
                                                                      scale=SCALE, wflag=True, pflag=False)

        ass_matrix = ass_matrix.cuda().detach()
        weight_matrix = weight_matrix.cuda().detach()
        images_s = Variable(images_s).cuda(args.gpu)
        labels_s = Variable(labels_s.long()).cuda(args.gpu)

        gfeats_s, pred_source = model(images_s)
        pred_source_temp = F.softmax(pred_source, dim=1)
        loss_ot_i = torch.tensor(0)

        pred_source_temp = interp_pred(pred_source_temp)
        pred_target_temp = interp_pred(pred_target_temp)
        pred_shape = pred_source_temp.shape
        pred_source_temp = pred_source_temp.view(pred_shape[0], pred_shape[1], -1).transpose(2, 1)
        pred_target_temp = pred_target_temp.view(pred_shape[0], pred_shape[1], -1).transpose(2, 1)
        with torch.no_grad():
            row_ind, col_ind = assignment_for_arrays(pred_source_temp, pred_target_temp, wflag=False, pflag=False)
            out_ass_matrix = ass_vector2matrix(row_ind, col_ind, length=pred_shape[-2] * pred_shape[-1])
            out_ass_matrix = torch.tensor(out_ass_matrix).float().unsqueeze(0).to(pred_source_temp.device)

        gfeats_s_temp = F.softmax(interp_pred(gfeats_s), dim=1)
        gfeats_t_temp = F.softmax(interp_pred(gfeats_t), dim=1)
        gfeats_s_temp = gfeats_s_temp.view(gfeats_s.shape[0], gfeats_s.shape[1], -1).transpose(2, 1)
        gfeats_t_temp = gfeats_t_temp.view(gfeats_t.shape[0], gfeats_t.shape[1], -1).transpose(2, 1)
        weight_matrix = torch.tensor(weight_matrix).unsqueeze(0).to(pred_source_temp.device)
        loss_ot_f, _ = assignc_w(gfeats_s_temp, gfeats_t_temp, ass_matrix, weight_matrix)
        loss_ot_o, _ = assignc_w(pred_source_temp, pred_target_temp, out_ass_matrix, weight_matrix)

        pred_source = interp_source(pred_source)

        loss_seg = seg_criterion(pred_source, labels_s)
        loss = loss_seg + loss_ot_o + loss_ot_f

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i_iter % 100 == 0:
            print('exp = {}'.format(args.snapshot_dir))
            print(
                'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f}'.format(
                    i_iter, args.num_steps, loss_seg))
            print(
                'iter = {0:6d}/{1:6d}, loss_ot_i = {2:.4f}, loss_ot_f = {3:.4f}, loss_ot_o = {4:.4f}'.format(
                    i_iter, args.num_steps, loss_ot_i, loss_ot_f, loss_ot_o))

        f_loss = open(osp.join(args.snapshot_dir, 'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f}\n'.format(
            loss_seg, loss_ot_i, loss_ot_f, loss_ot_o))
        f_loss.close()

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'P-V_' + str(args.num_steps) + '.pth'))
            uavidtesting(args, model, overlap=1 / 5)
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'P-V_' + str(i_iter) + '.pth'))
            uavidtesting(args, model, overlap=1 / 5, iter=i_iter)

if __name__ == '__main__':
    main()




