# 将mask转化为二值的png
import os
from PIL import Image
from tqdm import tqdm

# path =r'/exstorage/xyc/Mask_RCNN-master/groupdata-fire/data_processed/fire/mask'#标签原始路径
path ='/exstorage/xyc/Mask_RCNN-master/groupdata-fire/data_processed/fire/mask'#标签原始路径
save ='/exstorage/xyc/Mask_RCNN-master/unet-pytorch-main/VOCdevkit/VOC2007/SegmentationClass'#标签保存路径

list = os.listdir(path)
list2 = []

for i in tqdm(list):
    new = os.path.join(path,i)
    img = Image.open(new)
    w,h = img.size[0],img.size[1]
    for a in range(w):
        for b in range(h):
            c=  img.getpixel((a,b))
            if c !=0:
                img.putpixel((a,b),1)
    save_ = os.path.join(save,i)
    img.save(save_.replace('jpg','png'))