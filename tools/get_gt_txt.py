#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

image_ids = open('../../datasets/tt100k/test_new/ids.txt').read().strip().split()
annos = json.loads(open('../../datasets/tt100k/test_new/annotations.json').read())
clses = open('../model_data/tt100k_classes.txt').read().splitlines()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in tqdm(image_ids):
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        img = annos['imgs'][image_id]
        objs = img['objects']

        for obj in objs:
            obj_name = obj['category']
            if obj_name in clses:
                bbox = obj['bbox']
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])
                b = (xmin, ymin, xmax, ymax)
                new_f.write("%s %s %s %s %s\n" % (obj_name, xmin, ymin, xmax, ymax))
print("Conversion completed!")
