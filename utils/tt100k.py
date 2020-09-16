import os
import json

class TT100K():
    def __init__(self, data_path):
        self.data_path = data_path
        self.annotations_path = os.path.join(data_path, 'annotations.json')
        self.train_path = os.path.join(data_path, 'train')
        self.test_path = os.path.join(data_path, 'test')

        self.annotations = json.loads(open(self.annotations_path).read())
        self.types = self.annotations['types']
        self.ids_train = open(os.path.join(self.train_path, 'ids.txt')).read().splitlines()
        self.ids_test = open(os.path.join(self.test_path, 'ids.txt')).read().splitlines()

    def get_image_annotation(self, image_id):
        return self.annotations['imgs'][image_id]

    def get_image_path(self, image_annotation):
        return image_annotation['path']

    def get_label_from_anno(self, image_annotation):
        labels = []
        for obj in image_annotation['objects']:
            category = obj['category']
            category_id = self.types.index(category)
            xmin = int(obj['bbox']['xmin'])
            ymin = int(obj['bbox']['ymin'])
            xmax = int(obj['bbox']['xmax'])
            ymax = int(obj['bbox']['ymax'])
            label = [xmin,ymin,xmax,ymax,category_id]
            labels.append(label)
        return labels

    def get_label_from_id(self, image_id):
        image_annotation = self.get_image_annotation(image_id)
        label = self.get_label_from_anno(image_annotation)
        return label

if __name__ == '__main__':
    data_path = '../../datasets/tt100k'
    tt100k = TT100K(data_path)
    img_annotation = tt100k.get_image_annotation('10056')
    img_path = tt100k.get_image_path(img_annotation)
    label = tt100k.get_label_from_id('10056')
    print(img_annotation)
