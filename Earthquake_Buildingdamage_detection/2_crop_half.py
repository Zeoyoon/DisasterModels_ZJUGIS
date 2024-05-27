# 将 1024 的图片切片成 512 或 256
# 数千张照片裁切，需要 python 多线程

from osgeo import osr, gdal
import numpy as np
import os
from PIL import Image
import time
from tqdm import tqdm
import joblib
from skimage import io


def get_file_names(data_dir, file_type=None):
    if file_type is None:
        file_type = ['tif', 'tiff']
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):  # disdir 为灾害事件的文件夹
        for filename in file_name_list:
            apath = maindir + '/' + filename
            apath = apath.replace("\\", "/")
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name


def get_same_img(img_dir, img_name):
    result = {}
    for idx, name in enumerate(img_name):
        temp_name = ''
        for idx2, item in enumerate(name.split('_')[:-4]):
            if idx2 == 0:
                temp_name = temp_name + item
            else:
                temp_name = temp_name + '_' + item

        if temp_name in result:
            result[temp_name].append(img_dir[idx])
        else:
            result[temp_name] = []
            result[temp_name].append(img_dir[idx])
    return result


def cut_png(data_dir_list, fold, out_size=512):
    fold_len = len(data_dir_list) // NFOLD
    start_idx = fold_len * fold
    end_idx = fold_len * (fold + 1)

    if fold == NFOLD - 1:
        end_idx = NUM

    for each_dir in data_dir_list[start_idx:end_idx]:
        image = gdal.Open(each_dir).ReadAsArray()[:3]
        out_type = 'png'

        # 对于单波段的 mask 图像
        if image.shape[0] != 3:
            image = image[np.newaxis, :, :]

        cut_factor_row = int(np.ceil(image.shape[1] / out_size))
        cut_factor_clo = int(np.ceil(image.shape[2] / out_size))
        for i in range(cut_factor_row):
            for j in range(cut_factor_clo):

                if i == cut_factor_row - 1:
                    i = int(image.shape[1] / out_size - 1)
                else:
                    pass

                    if j == cut_factor_clo - 1:
                        j = int(image.shape[2] / out_size - 1)
                    else:
                        pass

                start_x = int(np.rint(i * out_size))
                start_y = int(np.rint(j * out_size))
                end_x = int(np.rint((i + 1) * out_size))
                end_y = int(np.rint((j + 1) * out_size))

                temp_image = image[:, start_x:end_x, start_y:end_y]
                # temp_image = image[start_x:end_x, start_y:end_y, :]

                if temp_image.shape[1] == 0 or temp_image.shape[2] == 0:
                    continue

                # if temp_image[0][0][0] == 255 and temp_image[0][0][out_size-1] == 255 \
                #     and temp_image[0][out_size-1][0] == 255 and temp_image[0][out_size-1][out_size-1] == 255:
                #     continue

                if temp_image.shape[1] < out_size or temp_image.shape[2] < out_size:
                    padx = int(out_size - temp_image.shape[1])
                    pady = int(out_size - temp_image.shape[2])
                    temp_image = np.pad(temp_image, ((0, 0), (0, padx), (0, pady)), 'constant',
                                        constant_values=((0, 0), (0, 255), (0, 255)))

                print('temp_image:', temp_image.shape)

                out_dir_images = out_dir + '/' + each_dir.split('/')[-1].split('.')[0] + '_#' + str(int(i)) + '_' + str(
                    int(j)) + '.' + out_type

                out_image = temp_image.transpose(1, 2, 0)
                out_image = np.squeeze(out_image)
                out_image = Image.fromarray(np.uint8(out_image))

                out_image.save(out_dir_images)


def cut(in_dir, out_dir, file_type=None, out_size=512):
    out_type = 'png'
    if file_type is None:
        file_type = ['png']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_dir_list, _ = get_file_names(in_dir, file_type)
    # count = 0
    print('Cut begining for ', str(len(data_dir_list)), ' images.....')

    ### 多线程切片
    global NFOLD
    global NUM
    NUM = len(data_dir_list)
    fold_len = NUM // NFOLD

    time_start = time.time()

    # _ = joblib.Parallel(n_jobs=NFOLD)(
    #     joblib.delayed(cut_png)(data_dir_list, fold, out_size) for fold in tqdm(range(NFOLD))
    # )
    for fold in tqdm(range(NFOLD)):
        cut_png(data_dir_list, fold, out_size)
    print('End of ' + str(len(data_dir_list)) + '/' + str(len(data_dir_list)) + '...')
    time_end = time.time()
    print('Time cost: ', time_end - time_start)
    print('Cut Finsh!')
    return 0

NFOLD = 10


if __name__ == '__main__':
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    ##### cut
    data_dir = '/exstorage/wcy/transfer/3-YOLO/data/VOCdevkit/VOC2007/JPEGImages0'
    out_dir = '/exstorage/wcy/transfer/3-YOLO/data/VOCdevkit/VOC2007/JPEGImages'
    file_type = ['png']
    out_type = 'png'
    cut_size = 416

    cut(data_dir, out_dir, file_type, cut_size)
