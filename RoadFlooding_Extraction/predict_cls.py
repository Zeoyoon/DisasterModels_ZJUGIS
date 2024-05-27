# 1. resize input images to 1300x1300
# 2. hrnet_cls model
import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import timeit
import cv2

#from zoo.models import Res34_Unet_Loc

from imgaug import augmenters as iaa


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import gc


from hrnet_ocr.config import update_config
from hrnet_ocr.seg_hrnet_ocr import get_cls_model



cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import argparse
parser = argparse.ArgumentParser(description='Run road Damage Model')
parser.add_argument('--ImgDir_input', type=str, default="./data/images/", help='the image input directory')

args = parser.parse_args()


test_dir = args.ImgDir_input
#test_dir = '../unlabel_data/xbd-midwest-flooding/images'
models_folder = 'weights'
pred_folder = args.ImgDir_input.replace('images', 'predict')
os.makedirs(pred_folder, exist_ok=True)


def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x

def combine_output(out1, out2, out3, out4, size):
    h_size = size//2   # half_size
    out1 = out1[:,:,:h_size,:h_size]
    out2 = out2[:,:,:h_size,-h_size:]
    out3 = out3[:,:,-h_size:,:h_size]
    out4 = out4[:,:,-h_size:,-h_size:]

    out = torch.cat((torch.cat((out1,out2),3), (torch.cat((out3,out4),3))), 2)
    return out
    

   
if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(pred_folder, exist_ok=True)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    # cudnn.benchmark = True

    snap_to_load = "HRNetOCR_cls_650_seed0_0911_best"
    cfg = update_config('hrnet_ocr/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
    model = get_cls_model(cfg)

    model = nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    model.eval()


    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' in f:
                
                fn = path.join(test_dir, f)
                fn_post = path.join(test_dir, f.replace('_pre_','_post_'))
                img_pre = cv2.imread(fn, cv2.IMREAD_COLOR)
                img_pre = img_pre.transpose(2, 0, 1)
                img_post = cv2.imread(fn_post, cv2.IMREAD_COLOR)
                img_post = img_post.transpose(2, 0, 1)
                img = np.concatenate([img_pre, img_post], axis=0)
                img = preprocess_inputs(img)
                img = torch.from_numpy(img[np.newaxis,:]).float()

                #img = torch.from_numpy(img.transpose((0, 3, 1, 2))).float()
                img = F.interpolate(input=img, size=(1300, 1300), mode='bilinear', align_corners=False)

                # split 1024*1024 into 4*512*512
                img1 = img[:,:,:650,:650]
                img2 = img[:,:,:650,650:]
                img3 = img[:,:,650:,:650]
                img4 = img[:,:,650:,650:]
                
                out1 = model(img1)
                out2 = model(img2)
                out3 = model(img3)
                out4 = model(img4)
                
                out1 = F.interpolate(input=out1, size=(512, 512), mode='bilinear', align_corners=False)
                out2 = F.interpolate(input=out2, size=(512, 512), mode='bilinear', align_corners=False)
                out3 = F.interpolate(input=out3, size=(512, 512), mode='bilinear', align_corners=False)
                out4 = F.interpolate(input=out4, size=(512, 512), mode='bilinear', align_corners=False)
                size = 1024

                out = torch.cat((torch.cat((out1,out2),3), (torch.cat((out3,out4),3))), 2)
                #out = combine_output(out1, out2, out3, out4, size)
                # msk_pred = torch.sigmoid(out[:, 0, ...]).cpu().numpy()
                
                # for j in range(msks.shape[0]):
                #     dices0.append(dice(msks[j, 0], msk_pred[j] > _thr))
                pre_mask = nn.Softmax(dim=1)(out).cpu().numpy()
                pre_mask = np.argmax(pre_mask, axis=1)
                msk = pre_mask # *127
                msk = msk.astype('uint8').transpose(1, 2, 0)
                cv2.imwrite(path.join(pred_folder, '{0}'.format(f.replace('.png', '.png'))), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
