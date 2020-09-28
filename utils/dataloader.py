from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from nets.yolo_training import Generator
import cv2
from utils.augment import DataAugmentForObjectDetection
import random


class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, hgain=.1, sgain=.5, vgain=.5):
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        # 用opencv读取图片并预处理
        image = cv2.imread(line[0])
        targets = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])  # xyxy,cls 可能为空shape = (0,)

        n = len(targets)
        if n:
            bboxes = targets[:, :4].reshape(n, 4)
            cls = targets[:, -1].reshape(n, 1)
            # 1、随机裁剪，更新image、bboxes和targets
            image, bboxes = random_crop_box(image, bboxes)
            targets = np.concatenate((cls, bboxes), axis=1)

        # 2、letterbox，输出416x416
        image, targets = letterbox(image, targets, new_shape=input_shape)

        # 3、随机透视变换
        image, targets = random_perspective(image, targets)

        # 4、色域变换
        augment_hsv(image, hgain=hgain, sgain=sgain, vgain=vgain)

        # 5、targets由cls,xyxy转为xyxy,cls
        targets = switch_targets(targets, format=0)

        return image, targets

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches  # 总共有多少行
        index = index % n
        img, y = self.get_random_data(lines[index], self.image_size[0:2])
        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))  # pytorch中通道C在前面，(C,H,W)
        tmp_targets = np.array(y, dtype=np.float32)  # 如果y是[]的话直接转换为空array
        return tmp_inp, tmp_targets  # targets可能为空array


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


def random_crop(image, targets=(), shape=(416, 416), wh_thr=2, ar_thr=20, area_thr=0.1):
    """
    以固定大小随机截取图像
    targets:cls,xyxy
    """
    oh, ow, _ = image.shape
    nh, nw = shape
    max_top, max_left = max(0, oh - nh), max(0, ow - nw)
    ymin = int(random.uniform(0, max_top))
    xmin = int(random.uniform(0, max_left))
    ymax = ymin + nh
    xmax = xmin + nw
    image = image[ymin:ymax, xmin:xmax]

    # 处理bboxes
    if len(targets):
        bboxes = targets[:, 1:5]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - ymin
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, nw)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, nh)
        i = box_candidates(box1=targets[:, 1:5].T, box2=bboxes.T, wh_thr=wh_thr, ar_thr=ar_thr, area_thr=area_thr)
        targets = targets[i]
        targets[:, 1:5] = bboxes[i]

    return image, targets


def random_crop_box(image, bboxes):
    """
    裁剪后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个array,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        crop_img:裁剪后的图像array
        crop_bboxes:裁剪后的bounding box的坐标array
    """
    # ---------------------- 裁剪图像 ----------------------
    if len(bboxes):
        h, w, _ = image.shape
        # 找到刚好包含所有box的框
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]  # 包含所有目标框的最小框到左边的距离
        max_u_trans = max_bbox[1]  # 包含所有目标框的最小框到顶端的距离
        max_r_trans = w - max_bbox[2]  # 包含所有目标框的最小框到右边的距离
        max_d_trans = h - max_bbox[3]  # 包含所有目标框的最小框到底部的距离

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes


def letterbox(img, targets=(), new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if len(targets):
        # targets:cls,xyxy
        box = targets[:, 1:5]
        box[:, 0] = box[:, 0] * r + dw
        box[:, 1] = box[:, 1] * r + dh
        box[:, 2] = box[:, 2] * r + dw
        box[:, 3] = box[:, 3] * r + dh

        i = box_candidates(targets[:, 1:5].T * r, box.T)
        targets = targets[i]
        targets[:, 1:5] = box[i]

    return img, targets


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def switch_targets(targets=(), format=0):
    """
    转换targets
    输入：targets:cls,xyxy或是xyxy,cls
        format:0-cls,xyxy to xyxy,cls
               1-xyxy,cls to cls,xyxy
    输出：xyxy,cls或是cls,xyxy
    """
    n = len(targets)
    if n:
        if format == 0:
            cls = targets[:, 0].reshape(n, 1)
            bboxes = targets[:, 1:5].reshape(n, 4)
            targets = np.concatenate((bboxes, cls), axis=1)
        elif format == 1:
            cls = targets[:, 4].reshape(n, 1)
            bboxes = targets[:, 0:4].reshape(n, 4)
            targets = np.concatenate((cls, bboxes), axis=1)
    return targets
