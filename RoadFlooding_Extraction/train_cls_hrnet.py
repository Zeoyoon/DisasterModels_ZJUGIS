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
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.cuda import amp

from torch.optim.adamw import AdamW
from losses.losses_basic import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from hrnet_ocr.seg_hrnet_ocr import HighResolutionNet_Roadcls

from imgaug import augmenters as iaa

from utils import *

from skimage.morphology import square, dilation

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from losses.sn8.sn8_loss import BuildingRoadFloodLoss, get_confusion_matrix, get_iou, get_pix_acc
import gc


from hrnet_ocr.config import update_config
from hrnet_ocr.seg_hrnet_ocr import get_cls_model


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

num_classes = 3
train_dirs = ['/home/wzy/road/data/train/hurricane_delta_ian/']

#train_dirs = ['../DAHiTra-main/data/xbd/train/', '../DAHiTra-main/data/xbd/tier3'] # , '../DAHiTra-main/data/xbd/tier3'
models_folder = 'weights'


input_shape = (650, 650)


all_files = []
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'pre'))):
        
        all_files.append(path.join(d, 'pre', f))


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('/pre/', '/post/').replace('_pre_', '_post_'), cv2.IMREAD_COLOR)
        if img.shape[:2] != (1300, 1300):
            img = cv2.resize(img, (1300, 1300), interpolation=cv2.INTER_LINEAR)
        if img2.shape[:2] != (1300, 1300):
            img2 = cv2.resize(img2, (1300, 1300), interpolation=cv2.INTER_LINEAR)

        msk0 = cv2.imread(fn.replace('/pre/', '/road_flood/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        if msk0.shape[:2] != (1300, 1300):
            msk0 = cv2.resize(msk0, (1300, 1300), interpolation=cv2.INTER_LINEAR)

        msk0[msk0>0] = 255
        lbl_msk1 = cv2.imread(fn.replace('/pre/', '/road_flood/').replace('_pre_', '_post_'), cv2.IMREAD_UNCHANGED)
        if lbl_msk1.shape[:2] != (1300, 1300):
            lbl_msk1 = cv2.resize(lbl_msk1, (1300, 1300), interpolation=cv2.INTER_LINEAR)



        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255

        if random.random() > 0.5:
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                    
        if random.random() > 0.8:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            
        if random.random() > 0.2:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.1:
            crop_size = random.randint(int(input_shape[0] / 1.15), int(input_shape[0] / 0.85))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 10)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk2[y0:y0+crop_size, x0:x0+crop_size].sum() * 4 
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]
        msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size]
        
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
            

        if random.random() > 0.96:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.96:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.96:
            img2 = change_hsv(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.9:
            if random.random() > 0.96:
                img = clahe(img)
            elif random.random() > 0.96:
                img = gauss_noise(img)
            elif random.random() > 0.96:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.96:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.9:
            if random.random() > 0.96:
                img2 = clahe(img2)
            elif random.random() > 0.96:
                img2 = gauss_noise(img2)
            elif random.random() > 0.96:
                img2 = cv2.blur(img2, (3, 3))
        elif random.random() > 0.9:
            if random.random() > 0.96:
                img2 = saturation(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img2 = brightness(img2, 0.9 + random.random() * 0.2)
            elif random.random() > 0.96:
                img2 = contrast(img2, 0.9 + random.random() * 0.2)

                
        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        if random.random() > 0.96:
            el_det = self.elastic.to_deterministic()
            img2 = el_det.augment_image(img2)

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2], axis=2)
        msk = (msk > 127)

        msk[..., 0] = True
        msk[..., 1] = dilation(msk[..., 1], square(5))
        msk[..., 2] = dilation(msk[..., 2], square(5))
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = False
        msk = msk * 1

        lbl_msk = msk.argmax(axis=2)

        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('/pre/', '/post/').replace('_pre_', '_post_'), cv2.IMREAD_COLOR)
        if img.shape[:2] != (1300, 1300):
            img = cv2.resize(img, (1300, 1300), interpolation=cv2.INTER_LINEAR)
        if img2.shape[:2] != (1300, 1300):
            img2 = cv2.resize(img2, (1300, 1300), interpolation=cv2.INTER_LINEAR)
        #print(fn.split('/')[-1].replace('.png', '_part1.png'))
        #msk_loc = cv2.imread(path.join(loc_folder, '{0}'.format(fn.split('/')[-1].replace('.png', '_part1.png'))), cv2.IMREAD_UNCHANGED) > 140#(0.3*255)

        
        lbl_msk1 = cv2.imread(fn.replace('/pre/', '/road_flood/').replace('_pre_','_post_'), cv2.IMREAD_UNCHANGED)
        if lbl_msk1.shape[:2] != (1300, 1300):
            lbl_msk1 = cv2.resize(lbl_msk1, (1300, 1300), interpolation=cv2.INTER_LINEAR)

        msk0 = np.zeros_like(lbl_msk1)
        msk1 = np.zeros_like(lbl_msk1)
        msk2 = np.zeros_like(lbl_msk1)
        msk0[lbl_msk1 == 0] = 255
        msk1[lbl_msk1 == 1] = 255
        msk2[lbl_msk1 == 2] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2], axis=2)
        
        msk = (msk > 127)

        msk = msk * 1

        lbl_msk = lbl_msk1[..., np.newaxis] 
        
        img = np.concatenate([img, img2], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()
        lbl_msk = torch.from_numpy(lbl_msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'lbl_msk': lbl_msk, 'fn': fn } #, 'msk_loc': msk_loc}  # msk_loc作为辅助评估分类效果的指标
        return sample


def validate(net, data_loader):
    # dices0 = []

    # tp = np.zeros((5,))
    # fp = np.zeros((5,))
    # fn = np.zeros((5,))

    # _thr = 0.3
    flood_confusion_matrix = np.zeros((3,3))
    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()   # one-hot 形式的msks
            lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)  # 单波段作为结果的msk
            imgs = sample["img"].float().cuda(non_blocking=True)
            # msk_loc = sample["msk_loc"].numpy() * 1

            img1 = imgs[:,:,:650,:650]
            img2 = imgs[:,:,:650,650:]
            img3 = imgs[:,:,650:,:650]
            img4 = imgs[:,:,650:,650:]
            
            out1 = model(img1)
            out2 = model(img2)
            out3 = model(img3)
            out4 = model(img4)
            out = torch.cat((torch.cat((out1,out2),3), (torch.cat((out3,out4),3))), 2)
            #msk_damage_pred = msk_pred[:, 1:, ...]   # 对于道路受影响/不受影影响
            
            flood_preds = F.interpolate(input=out, size=(1300, 1300), mode='bilinear', align_corners=False)  # (B, 5, H, W) 

            flood_preds = F.softmax(flood_preds, dim=1)  
            flood_preds = flood_preds.argmax(dim=1)  # (B, H, W)

            # flood
            b, _, h, w = lbl_msk.shape
            
            size = imgs.size()
            flood_targets = lbl_msk.long()   # (B, H, W)
            #flood_targets = torch.cat([torch.zeros(b,1,h,w).to(device), flood_labels], dim=1).argmax(dim=1).long() # (B, H, W)
            flood_confusion_matrix += get_confusion_matrix(flood_targets, flood_preds, size, 3)
            # flood    
        flood_pix_acc = np.diag(flood_confusion_matrix).sum() / flood_confusion_matrix.sum()
        flood_acc_cls = np.diag(flood_confusion_matrix) / flood_confusion_matrix.sum(axis=1)
        flood_acc_cls = np.nanmean(flood_acc_cls)
        divisor = flood_confusion_matrix.sum(axis=1) + flood_confusion_matrix.sum(axis=0) - \
                        np.diag(flood_confusion_matrix)
        flood_class_iou = np.diag(flood_confusion_matrix) / divisor
        flood_mean_iou = np.nansum(flood_class_iou[1:]) / 2  # 仅考虑后两个class
    return flood_mean_iou
    #return flood_pix_acc, flood_mean_iou


def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch(current_epoch, seg_loss, ce_loss, model, optimizer, scheduler, scaler, train_data_loader):
    losses = AverageMeter()
    losses1 = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        lbl_msk = sample["lbl_msk"].cuda(non_blocking=True)
        with amp.autocast(enabled=True):    
            out = model(imgs)

            out = F.interpolate(input=out, size=input_shape, mode='bilinear', align_corners=False)  # (B, 3, H, W) 

            loss0 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
            loss1 = seg_loss(out[:, 1, ...], msks[:, 1, ...])
            loss2 = seg_loss(out[:, 2, ...], msks[:, 2, ...])

            loss3 = ce_loss(out, lbl_msk)

            loss = 0.2* loss0 + 3* loss1 + 6 * loss2  
        #loss = loss5
        with torch.no_grad():
            _probs = 1 - torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, 1 - msks[:, 0, ...])

        scaler.scale(loss).backward()


        scaler.step(optimizer)  # optimizer.step
        scaler.update()
        optimizer.zero_grad()

        losses.update(loss.item(), imgs.size(0))
        losses1.update(loss3.item(), imgs.size(0))

        dices.update(dice_sc, imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); cce_loss {loss1.val:.4f} ({loss1.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))
        
        
        #loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.999)
        optimizer.step()

    scheduler.step(current_epoch)

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; CCE_loss {loss1.avg:.4f}; Dice {dice.avg:.4f}".format(
            current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, dice=dices))


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    
    seed = 2
    # seed = int(sys.argv[1])
    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'

    cudnn.benchmark = True

    batch_size = 4
    val_batch_size = 4
    
    snapshot_name = "HRNetOCR_cls_650_seed{}_0911".format(seed)   # 0911 for hurricane ian and delta

    file_classes = []

    train_idxs, val_idxs = train_test_split(np.arange(len(all_files)), test_size=0.1, random_state=seed)

    np.random.seed(seed + 545)
    random.seed(seed + 545)

    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=6, shuffle=True, pin_memory=False, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=6, shuffle=False, pin_memory=False)
    
    cfg = update_config('hrnet_ocr/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
    model = get_cls_model(cfg)
    
    params = model.parameters()

    # optimizer = torch.optim.Adam(params, lr=0.00001, betas=(0.9,0.999),eps=1e08,weight_decay=1e-6)
    optimizer = AdamW(params, lr=0.00015, weight_decay=1e-6)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)

    # model, optimizer = amp.initialize(model.to(device), optimizer, opt_level="O1")   # O1
    

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 11, 17, 23, 29, 33, 47, 50, 60, 70, 90, 110, 130, 150, 170, 180, 190], gamma=0.5)
    
    scaler = amp.GradScaler(enabled=True)
    
    model = nn.DataParallel(model).cuda()
    
    #snap_to_load = "SegFormer_cls_{}_512_seed{}_best".format(phi, seed)
    
    snap_to_load = "HRNetOCR_loc_650_seed0_tune_best"  #.format(seed)
    
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)


    best_score = checkpoint['best_score']
    best_score = 0
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
            .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    del loaded_dict
    del sd
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    seg_loss = ComboLoss({'dice': 1.0, 'focal': 2.0}, per_image=False).cuda()   #0.5, 2.0
    ce_loss = nn.CrossEntropyLoss().cuda()

    torch.cuda.empty_cache()
    for epoch in range(35):
        # if epoch == 0:
        #     best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)   # best iou
        torch.cuda.empty_cache()
        train_epoch(epoch, seg_loss, ce_loss, model, optimizer, scheduler, scaler, train_data_loader)

        torch.cuda.empty_cache()
        best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
