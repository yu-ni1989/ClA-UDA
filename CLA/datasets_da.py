#############################################################################################
### Created by Huan Ni ######################################################################
#############################################################################################
import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
# from ParsingModels.Datasets.UAVid.colorTransformer_a import PotsdamColorTransformer
from DA.CLA.ColorTransformer_da import ColorTransformer


class UAVidTrainWithScale(data.Dataset):
    def __init__(self, root, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=None, mirr=False, ignore_label=255, subfold='train', mark=None):
        # pos1 = list_path.rfind('/')
        # self.root = list_path[0:pos1]
        # self.list_path = list_path
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirr
        self.subfold = subfold
        self.mark = mark
        self.colormap = ColorTransformer(mode='potsdam')

        self.img_ids = []
        for subfold in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, subfold)):
                image_path = self.root + '/' + subfold + '/' + 'Images'
                # label_path = self.root + '/' + subfold + '/' + 'TrainId'
                names = [name for name in os.listdir(image_path) if name[0] != '.']
                self.img_ids = self.img_ids + [image_path + '/' + name for name in names]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for item in self.img_ids:
            name = item[item.rfind('/') + 1 : ]
            img_file = item
            label_file = item.replace('Images', 'TrainId')
            label_file = label_file.replace('png', 'bmp')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.7 + random.randint(0, 14) / 10.0
        f_scale = self.scale[0] + random.randint(0, self.scale[1]) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        # print(datafiles['name'])
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # image = image[:, :, ::-1]
        label = cv2.imread(datafiles["label"], cv2.IMREAD_UNCHANGED)
        # label = label[:, :, ::-1]
        # label = self.colormap.transform(label)

        size = image.shape
        name = datafiles["name"]
        if self.scale != None:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class UAVidVal(data.Dataset):
    def __init__(self, root, max_iters=None, crop_size=(512, 512), mean=(0, 0, 0), ignore_label=255):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        # self.colormap = PotsdamColorTransformer()

        self.img_ids = []
        for subfold in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, subfold)):
                image_path = self.root + '/' + subfold + '/' + 'Images'
                # label_path = self.root + '/' + subfold + '/' + 'TrainId'
                names = [name for name in os.listdir(image_path) if name[0] != '.']
                self.img_ids = self.img_ids + [image_path + '/' + name for name in names]

        self.files = []
        for item in self.img_ids:
            name = item[item.rfind('/') + 1 : ]
            seq = item[item.find('seq') : item.find('/Images')]
            img_file = item
            label_file = item.replace('Images', 'TrainId')
            label_file = label_file.replace('png', 'bmp')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "seq": seq
            })

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[:, :, ::-1]
        label = cv2.imread(datafiles["label"], cv2.IMREAD_UNCHANGED)

        size = image.shape
        name = datafiles["name"]
        seq = datafiles["seq"]
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        image -= self.mean
        img_h, img_w = label.shape

        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name, seq

class PotsdamDatasetWithScale(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=None, mirr=False, ignore_label=255, subfold='train', mark=None):
        pos1 = list_path.rfind('/')
        self.root = list_path[0:pos1]
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirr
        self.subfold = subfold
        self.mark = mark
        self.colormap = ColorTransformer(mode='potsdam')

        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for item in self.img_ids:
            name = item[0]
            name_label = name.replace(self.mark, 'label')
            img_file = osp.join(self.root, f"images/{self.subfold}/%s" % name)
            label_file = osp.join(self.root, f"labels_RGB/{self.subfold}/%s" % name_label)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        # f_scale = 0.7 + random.randint(0, 14) / 10.0
        f_scale = self.scale[0] + random.randint(0, self.scale[1]) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        # print(datafiles['name'])
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # image = image[:, :, ::-1]
        label = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
        label = label[:, :, ::-1]
        label = self.colormap.transform(label)

        size = image.shape
        name = datafiles["name"]
        if self.scale != None:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

