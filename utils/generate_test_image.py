import json
import os
import cv2
from utils import augment

if __name__ == '__main__':
    data_path = '../../datasets/tt100k'
    test_path = os.path.join(data_path, 'test')
    test_new_path = os.path.join(data_path, 'test_new')
    annotations_path = os.path.join(data_path, 'annotations.json')
    ids = open(os.path.join(test_path, 'ids.txt')).read().splitlines()
    annos = json.loads(open(annotations_path).read())
    dataAug = augment.DataAugmentForObjectDetection()

    for img_id in ids:
        # 对于每一个图像，拿到该图像的bboxes
        bboxes = []
        for obj in annos['imgs'][img_id]['objects']:
            xmin = obj['bbox']['xmin']
            ymin = obj['bbox']['ymin']
            xmax = obj['bbox']['xmax']
            ymax = obj['bbox']['ymax']
            bboxes.append([xmin, ymin, xmax, ymax])
        img_path = os.path.join(data_path, annos['imgs'][img_id]['path'])
        img = cv2.imread(img_path)
        auged_img, auged_bboxes = dataAug._crop_img_bboxes(img, bboxes)  # 得到增强后的图像和bboxes
        cv2.imwrite(os.path.join(test_new_path, img_id + '.jpg'),auged_img)  # 保存增强后的图像
        # 修改标签文件
        annos['imgs'][img_id]['path'] = 'test_new/' + img_id + '.jpg'  # 新的图像地址

        for i, obj in enumerate(annos['imgs'][img_id]['objects']):  # 新的bbox
            t = auged_bboxes[i]

            obj['bbox']['xmin'] = t[0]
            obj['bbox']['ymin'] = t[1]
            obj['bbox']['xmax'] = t[2]
            obj['bbox']['ymax'] = t[3]

    # 保存标签文件
    with open(os.path.join(test_new_path, 'annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(annos, f)

