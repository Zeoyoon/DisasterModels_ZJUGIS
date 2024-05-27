import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed, erosion
from skimage import io

from shapely.wkt import loads
from shapely.geometry import mapping, Polygon

# import matplotlib.pyplot as plt
# import seaborn as sns

import json

masks_dir = 'masks'

train_dirs = ['train', 'tier3']


def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    rect = cv2.minAreaRect(exteriors[0]) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    
    box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    hen=[]
    shu=[]
    for i in box:
        hen.append(i[0])
        shu.append(i[1])
    l=min(hen)
    r=max(hen)
    t=min(shu)
    b=max(shu)
    box = np.int0(box)
    # 画出来
    cv2.drawContours(img_mask, [box], 0, (255, 0, 0), 1)
    cv2.imwrite('contours.png', img_mask)

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 # ?
}


def process_image(json_file):
    js1 = json.load(open(json_file))
    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

    msk = np.zeros((1024, 1024), dtype='uint8')
    msk_damage = np.zeros((1024, 1024), dtype='uint8')

    for feat in js1['features']['xy']:
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = 255

    for feat in js2['features']['xy']:
        poly = loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype]

    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_pre_disaster.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_post_disaster.png'), msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])



if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    for d in train_dirs:
        makedirs(path.join(d, masks_dir), exist_ok=True)
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(d, 'labels', f.replace('_pre_disaster.png', '_pre_disaster.json')))


    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
