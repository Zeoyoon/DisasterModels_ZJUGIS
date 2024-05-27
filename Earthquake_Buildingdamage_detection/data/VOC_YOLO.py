# coding: utf-8
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 这里就体现出来了咱们在1.2步骤的时候我说的尽量按照那个目录名进行操作的优势，
# 在这可以剩下很多去修改名称的精力
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('val'), ('train'), ('test')]  # 我只用了VOC2007

classes = ["collapsed"]


# classes = ["face"]  # 修改为自己的label

def convert(size, box):
    dw = 1. / (size[0])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dw = 1./((size[0])+0.1)
    dh = 1. / (size[1])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dh = 1./((size[0])+0.1)
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id):
    in_file = open('./data/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id))
    out_file = open('./data/VOCdevkit/VOC2007/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

for image_set in sets:
    if not os.path.exists('VOCdevkit/VOC2007/labels/'):
        os.makedirs('VOCdevkit/VOC2007/labels/')
    image_ids = open('./data/VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/data/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n' % (wd,image_id))
        convert_annotation(2007, image_id)
    list_file.close()

# 这块是路径拼接，暂时用不上，先都注释了
# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")