import os

from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import argparse
parser = argparse.ArgumentParser(description='Run Building Damage Model')
parser.add_argument('--ImgDir_input', type=str, default="./data/images/", help='the image input directory')
parser.add_argument('--PredictDir_output', type=str, default="./data/predict/", help='the prediction output directory')

args = parser.parse_args()

test_dir = args.ImgDir_input
sub_folder = args.PredictDir_output

pred_folders = [test_dir.replace("images", "cls50_0"), test_dir.replace("images", "cls50_1"), test_dir.replace("images", "cls50_2")]
pred_coefs = [1.0] * 3
loc_folders = [test_dir.replace("images", "loc")]
loc_coefs = [1.0] * 1 


_thr = [0.38, 0.13, 0.14]

def process_image(f):
    preds = []
    _i = -1
    for d in pred_folders:
        _i += 1
        #print(path.join(d, f))
        try:
            msk1 = cv2.imread(path.join(d, f), cv2.IMREAD_UNCHANGED)
        except:
            print('msk1 error', path.join(d, f))
        try:
            msk2 = cv2.imread(path.join(d, f.replace('_part1', '_part2')), cv2.IMREAD_UNCHANGED)
        except:
            print('msk2 error', path.join(d, f.replace('_part1', '_part2')))
        msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
        preds.append(msk * pred_coefs[_i])
    preds = np.asarray(preds).astype('float').sum(axis=0) / np.sum(pred_coefs) / 255
    
    loc_preds = []
    _i = -1
    for d in loc_folders:
        _i += 1
        try:
            msk = cv2.imread(path.join(d, f), cv2.IMREAD_UNCHANGED)
        except:
            print('msk error',path.join(d, f))
        loc_preds.append(msk * loc_coefs[_i])
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / np.sum(loc_coefs) / 255

    loc_preds = loc_preds 

    msk_dmg = preds[..., 1:].argmax(axis=2) + 1
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | ((loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')
    
    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    cv2.imwrite(path.join(sub_folder, '{0}'.format(f.replace('_pre_', '_localization_').replace('_part1', '_prediction'))), msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(sub_folder, '{0}'.format(f.replace('_pre_', '_damage_').replace('_part1', '_prediction'))), msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(sub_folder, exist_ok=True)

    all_files = []
    for f in tqdm(sorted(listdir(pred_folders[0]))):
        if '_part1.png' in f:
            all_files.append(f)
    print(len(all_files))
    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
