# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection
# author:
#     maozezhong 2018-6-27
# https://github.com/maozezhong/CV_ToolBox/blob/master/DataAugForObjectDetection/DataAugmentForObejctDetection.py
##############################################################

# 包括:
#     1. 裁剪(需改变bbox)
#     2. 平移(需改变bbox)
#     3. 改变亮度
#     4. 加噪声
#     5. 旋转角度(需要改变bbox)
#     6. 镜像(需要改变bbox)
#     7. cutout
# 注意:
#     random.seed(),相同的seed,产生的随机数是一样的!!


import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure


def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=10,
                 crop_rate=1, shift_rate=0, change_light_rate=0.5,
                 add_noise_rate=0, flip_rate=0):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate

    # 加噪声
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time()))
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True) * 255

    # 调整亮度
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)


    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_AREA)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.triangular(0, d_to_left))
        crop_y_min = int(y_min - random.triangular(0, d_to_top))
        crop_x_max = int(x_max + random.triangular(0, d_to_right))
        crop_y_max = int(y_max + random.triangular(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:  # 0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转
            flip_img = cv2.flip(flip_img, 1)  # 1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

        return flip_img, flip_bboxes

    '''
    letterbox:将图片补成正方形并缩放到需要的大小
    输入：
        image:opencv的图像矩阵 (H,W,C) BGR
        bboxes：bbox矩阵 （m,4）每列为(xmin,ymin,xmax,ymax)
        new_shape:将图片调整为该尺寸
    输出：
        new_image
        new_bboxes
    '''

    def letterbox(self, image, bboxes, new_shape=(416, 416)):
        ih, iw = image.shape[0:2]
        w, h = new_shape
        scale = min(h / ih, w / iw)
        nh = int(ih * scale)
        nw = int(iw * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        top = (h - nh) // 2
        bottom = h - nh - top
        left = (w - nw) // 2
        right = w - nw - left
        new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        bboxes = np.array(bboxes)
        new_bboxes = bboxes*scale + np.array([left, top, left, top])
        new_bboxes = new_bboxes.astype(int)
        new_bboxes = new_bboxes.tolist()
        return new_image, new_bboxes

    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        # print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效
            if random.random() < self.crop_rate:  # 裁剪
                # print('裁剪')
                change_num += 1
                img, bboxes = self._crop_img_bboxes(img, bboxes)

            if random.random() > self.rotation_rate:  # 旋转
                # print('旋转')
                change_num += 1
                # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                angle = random.random() * self.max_rotation_angle * 2 - self.max_rotation_angle
                scale = random.uniform(0.7, 0.8)
                # print(angle)
                # print(scale)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if random.random() < self.shift_rate:  # 平移
                # print('平移')
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if random.random() > self.change_light_rate:  # 改变亮度
                # print('亮度')
                change_num += 1
                img = self._changeLight(img)

            if random.random() < self.add_noise_rate:  # 加噪声
                # print('加噪声')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  # cutout
                # print('cutout')
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                   threshold=self.cut_out_threshold)

            if random.random() < self.flip_rate:  # 翻转
                # print('翻转')
                change_num += 1
                img, bboxes = self._filp_pic_bboxes(img, bboxes)
            print('\n')
        # print('------')
        return img, bboxes


if __name__ == '__main__':

    ### test ###
    import json

    dataAug = DataAugmentForObjectDetection()

    image_path = '../../datasets/tt100k/test/2315.jpg'
    annotations_path = '../../datasets/tt100k/annotations.json'
    imgid = '2315'
    annos = json.loads(open(annotations_path).read())
    bboxes = []
    for obj in annos['imgs'][imgid]['objects']:
        xmin = obj['bbox']['xmin']
        ymin = obj['bbox']['ymin']
        xmax = obj['bbox']['xmax']
        ymax = obj['bbox']['ymax']
        bboxes.append([xmin, ymin, xmax, ymax])

    img = cv2.imread(image_path)
    # show_pic(img, bboxes)  # 原图

    auged_img, auged_bboxes = dataAug.dataAugment(img, bboxes)
    print(auged_img.shape)
    show_pic(auged_img, auged_bboxes)  # 强化后的图
