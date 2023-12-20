#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################

import argparse

# import torch
import torch.nn as nn
import torch.utils.data
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
# from ParsingModels.Datasets.datasets_g import RSTeaDataValSet
from DA.CLA.datasets_da import UAVidVal
from DA.Datasets.ColorTransformer_da_b import ColorTransformer

import torch.nn.functional as F

from math import ceil
from PIL import Image as PILImage
from datetime import datetime
from DA.CLA.label_tools import pseudo_label


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_sliding(args, G, F1, F2, image, overlap=2/3, mode='cuda'):
    classes = args.num_classes
    h, w = map(int, args.input_size.split(','))
    tile_size = (h, w)
    image_size = image.shape

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # if tile_counter % 20 == 0:
            #     print("Predicting tile %i" % tile_counter)

            padded_img = torch.from_numpy(padded_img)
            # padded_img = padded_img.to(G.device())
            if mode == 'cuda':
                padded_img = padded_img.cuda()

            ########## UDA ###############
            output = G(padded_img)
            output1 = F.softmax(F1(output), dim=1)
            output2 = F.softmax(F2(output), dim=1)
            padded_prediction = output1 + output2
            padded_prediction = F.interpolate(padded_prediction, padded_img.shape[-2:], mode='bilinear', align_corners=True)

            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction
            # # padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            padded_prediction = padded_prediction[0, :, :, :]
            padded_prediction = padded_prediction.cpu().data.numpy()
            padded_prediction = padded_prediction.transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions


    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs


def predict_whole(args, G, F1, F2, image):
    if args.gpu == True:
        image = image.cuda()
    interp = nn.Upsample(size=image.shape[-2:], mode='bilinear', align_corners=True)
    # prediction = net(image.cuda(), recurrence)
    ########## UDA ###############
    _, output, _, _ = G(image)
    output1 = F1(output)
    output2 = F2(output)
    prediction = output1 + output2
    # padded_prediction = F.interpolate(output_add, padded_img.shape[-2:], mode='bilinear', align_corners=True)

    if isinstance(prediction, list):
        prediction = prediction[0]
    prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    return prediction

def predict_whole_n(args, net, image):
    if args.gpu == True:
        image = image.cuda()
    prediction = net(image, y=None, labs=args.num_classes, ignore_label=args.ignore_label, pred=True)
    if isinstance(prediction, list):
        prediction = prediction[0]
    # prediction = interp(prediction).cpu().data[0].numpy().transpose(1, 2, 0)
    # prediction = prediction[0, :, :, :]
    # prediction = prediction.permute(1, 2, 0)
    prediction = prediction.cpu().data[0].numpy().transpose(1, 2, 0)
    return prediction


def predict_sliding_for_one(args, model, image, overlap=2/3, mode='cuda'):
    classes = args.num_classes
    h, w = map(int, args.input_size.split(','))
    tile_size = (h, w)
    image_size = image.shape

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # if tile_counter % 20 == 0:
            #     print("Predicting tile %i" % tile_counter)

            padded_img = torch.from_numpy(padded_img)
            # padded_img = padded_img.to(G.device())
            if mode == 'cuda':
                padded_img = padded_img.cuda()

            ########## UDA ###############
            _, padded_prediction = model(padded_img)
            padded_prediction = F.interpolate(padded_prediction, padded_img.shape[-2:], mode='bilinear', align_corners=True)

            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction
            # # padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            padded_prediction = padded_prediction[0, :, :, :]
            padded_prediction = padded_prediction.cpu().data.numpy()
            padded_prediction = padded_prediction.transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions


    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs


def predict_sliding_forone(args, G, image, overlap=2/3):
    classes = args.num_classes
    h, w = map(int, args.input_size_target.split(','))
    tile_size = (h, w)
    image_size = image.shape

    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    # print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((image_size[2], image_size[3], classes))
    count_predictions = np.zeros((image_size[2], image_size[3], classes))
    tile_counter = 0

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], image_size[3])
            y2 = min(y1 + tile_size[0], image_size[2])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = image[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            # if tile_counter % 20 == 0:
            #     print("Predicting tile %i" % tile_counter)

            padded_img = torch.from_numpy(padded_img)
            # padded_img = padded_img.to(G.device())
            padded_img = padded_img.cuda()

            ########## UDA ###############
            _, padded_prediction = G(padded_img)
            padded_prediction = F.interpolate(padded_prediction, padded_img.shape[-2:], mode='bilinear', align_corners=True)

            if isinstance(padded_prediction, list):
                padded_prediction = padded_prediction
            # # padded_prediction = interp(padded_prediction).cpu().data[0].numpy().transpose(1, 2, 0)
            padded_prediction = padded_prediction[0, :, :, :]
            padded_prediction = padded_prediction.cpu().data.numpy()
            padded_prediction = padded_prediction.transpose(1, 2, 0)
            prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions


    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs


def uavidtesting(args, model, overlap=2/3, iter=None):
    foldname = args.snapshot_dir + "/val"
    fa = None
    if iter != None:
        if not os.path.exists(args.snapshot_dir + "/val" + str(iter)):
            os.makedirs(args.snapshot_dir + "/val" + str(iter))
        fa = open(args.snapshot_dir + "/val" + str(iter) + "/accuracy_values.txt", 'w')
        foldname = args.snapshot_dir + "/val" + str(iter)
    else:
        if not os.path.exists(args.snapshot_dir + "/val"):
            os.makedirs(args.snapshot_dir + "/val")
        fa = open(args.snapshot_dir + "/val/accuracy_values.txt", 'w')

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    cudnn.enabled = True

    model.eval()

    valloader = data.DataLoader(
        UAVidVal(args.data_dir_test_target, mean=args.mean),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    clrEnc = ColorTransformer(mode='potsdam')
    dt = datetime.now()
    print('时间：(%Y-%m-%d %H:%M:%S %f): ', dt.strftime('%Y-%m-%d %H:%M:%S %f'))
    fa.write(str(dt.strftime('%Y-%m-%d %H:%M:%S %f')) + '\n')
    for index, batch in enumerate(valloader):
        if index > 20:
            break
        if index % 10 == 0:
            print('%d processd' % (index))
        image, label, size, name, seq = batch
        size = size[0].numpy()
        # image = F.softmax(image, dim=1)

        output = predict_sliding_for_one(args, model, image.numpy(), overlap=overlap, mode='cuda')
        plabel = pseudo_label(output)

        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        seg_pred_s = seg_pred[np.newaxis, :] # 1024 * 1024
        seg_pred_s = torch.from_numpy(seg_pred_s) # 1 * 1024 * 1024
        output_lab = PILImage.fromarray(plabel.astype(np.uint8))
        temp_name = seq[0] + '_' + name[0][0: name[0].rfind('.')]
        if not os.path.exists(foldname + '/Preds'):
            os.makedirs(foldname + '/Preds')
        output_lab.save(foldname + '/Preds/' + temp_name + '.png')

        seg_pred = clrEnc.inverse_transform(seg_pred)
        output_im = PILImage.fromarray(seg_pred)
        if not os.path.exists(foldname + '/PredImgs'):
            os.makedirs(foldname + '/PredImgs')
        output_im.save(foldname + '/PredImgs/' + temp_name + '.png')

    dt = datetime.now()
    print('时间：(%Y-%m-%d %H:%M:%S %f): ', dt.strftime('%Y-%m-%d %H:%M:%S %f'))
    fa.write(str(dt.strftime('%Y-%m-%d %H:%M:%S %f')) + '\n')


    return 1

