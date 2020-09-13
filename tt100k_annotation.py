import json

datadir = "../datasets/tt100k"
filedir = datadir + "/annotations.json"
ids = open(datadir + "/train/ids.txt").read().splitlines()
annos = json.loads(open(filedir).read())
classes = annos['types']
file = open('tt100k_train.txt', 'w')
with open('model_data/tt100k_classes.txt','w') as classes_file:
    for cls in classes:
        classes_file.write(cls+'\n')

for imgid in ids:
    img = annos['imgs'][imgid]
    imgpath = img['path']
    file.write(datadir+imgpath)
    objs = img['objects']
    for obj in objs:
        cls = obj['category']
        cls_id = classes.index(cls)
        bbox = obj['bbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])
        b = (xmin, ymin, xmax, ymax)
        file.write(" "+",".join([str(a) for a in b]) + ',' + str(cls_id))
    file.write('\n')
file.close()