import cv2
import random
import numpy as np


# 随机水平翻转
def random_horizontal_flip(image, bboxes):
    # random.random()方法返回一个随机数，其在0至1的范围之内
    if random.random() < 0.5:
        _, w, _ = image.shape
        # [::-1] 顺序相反操作
        # a = [1, 2, 3, 4, 5]
        # a[::-1]
        # Out[3]: [5, 4, 3, 2, 1]
        image = image[:, ::-1, :]
        # bboxes为m行4列，每行为一个bbox(xmin ,ymin ,xmax ,ymax)
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes


# 随机裁剪
def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        # 找到能把所有bbox都包含的最小框
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]  # 左边距
        max_u_trans = max_bbox[1]  # 上边距
        max_r_trans = w - max_bbox[2]  # 右边距
        max_d_trans = h - max_bbox[3]  # 下边距
        # 计算裁剪后的四个坐标点
        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes
