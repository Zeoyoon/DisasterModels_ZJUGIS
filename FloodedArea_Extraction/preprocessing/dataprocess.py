# -*- coding: utf-8 -*-
# @Time : 2022/8/21
# @Author : Miao Shen
# @File : cutimage.py


import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import torch
import logging
import rasterio
# from osgeo import gdal


def cut(in_file, size):
    """

    :param in_file: str
        path of source image
    :param size: int
        size of image

    :return: list
        a group of images in certain size
    """


    data = []
    ds=rasterio.open(in_file)
    image=ds.read()

    cut_factor_row = int(np.ceil(ds.height / size))  # 0
    cut_factor_clo = int(np.ceil(ds.width / size))  # 1
    count=ds.count


    for i in range(cut_factor_row):
        for j in range(cut_factor_clo):
            temp_image = np.zeros((count, size, size), dtype=float)

            if i == cut_factor_row - 1:
                start_x = int(np.rint(i * size))
                end_x = ds.height

            else:
                start_x = int(np.rint(i * size))
                end_x = int(np.rint((i + 1) * size))

            if j == cut_factor_clo - 1:
                start_y = int(np.rint(j * size))
                end_y = ds.width

            else:
                start_y = int(np.rint(j * size))
                end_y = int(np.rint((j + 1) * size))

            temp_image[:, 0:(end_x - start_x), 0:(end_y - start_y)] = image[:, start_x:end_x, start_y:end_y]



            # 本地测试出现影像边缘异常，需要进行替换
            # for k in range(0, len(temp_image)):
            #     s21, s22 = np.where(temp_image[k] < 0)
            #     # print(s21)
            #     for m in range(0, len(s21)):
            #         temp_image[k][s21[m]][s22[m]] = 0
            data.append(temp_image)

    return data,count

def flood_extraction_out(profile, img, output, size=512):
    """
    output the flood inundation map in study area

    """
    height = profile['height']
    width = profile['width']


    x = int(np.ceil(height / size)) - 1
    y = int(np.ceil(width / size)) - 1
    out = np.zeros(shape=(1, height, width), dtype=np.float32)


    for i in range(0, x + 1):
        for j in range(0, y + 1):

            if j == y and i == x:
                out[0][i * 512:height, j * 512:width] = img[i * (y + 1) + j][0:(height - x * 512),0:(width - y * 512)]

                continue
            if j == y:
                out[0][i * 512:(i + 1) * 512, j * 512:width] = img[i * (y + 1) + j][:, 0:(width - y * 512)]

                continue
            if i == x:
                out[0][i * 512:height, j * 512:(j + 1) * 512] = img[i * (y + 1) + j][0:(height - x * 512), :]

                continue
            a = img[i * (y + 1) + j]
            out[0][i * 512:(i + 1) * 512, j * 512:(j + 1) * 512] = a


    with rasterio.open(output, mode='w', **profile) as dst:
        dst.write(out)




class CutImage():
    """
    cut image
    """

    def __init__(self, s1_path,s1vh_path, s2_path,choice, SIZE=512):
        """

        :param s1_path: str
            path of Sentinel-1 image.
        :param s2_path: str
            path of Sentinel-2 image.
        :param choice: int
            select to use different images.
        :param SIZE: int
            size of Sentinel-1 and Sentinel-2 images
            .. note:: To improve the efficient of extraction, the original Sentinel-1 images and Sentinel-2 images will be cut into 512 pixels * 512 pixels.


        """


        super().__init__()
        self.s1_path = s1_path
        self.s1vh_path = s1vh_path
        self.s2_path = s2_path
        self.choice = choice
        self.size = SIZE


    def getprofile(self):
        image = rasterio.open(self.s1_path)
        profile = image.profile
        profile['count'] = 1
        return profile



    def run(self):
        """
        Get the result of image cutting
        :return: list
            a group of Sentinel-1 and Sentinel-2 images in certain size
        """


        logging.info('cut images')

        flood_data = []

        if self.choice == 1:
            if self.s1_path==self.s1vh_path:
                s1,count_s1 = cut(self.s1_path,self.size)
                for i in range(0, len(s1)):
                    arr_s1 = s1[i].reshape(count_s1, self.size, self.size)


                    arr_s1 = np.clip(arr_s1, -50, 1)
                    arr_s1 = (arr_s1 + 50) / 51

                    flood_data.append(arr_s1)
            else:
                s1_vv, count_s1 = cut(self.s1_path, self.size)
                s1_vh, count_s1 = cut(self.s1vh_path, self.size)
                for i in range(0, len(s1_vv)):
                    arr_s1 = np.zeros((2, 512, 512), dtype=float)
                    arr_s1[0, :, :] = s1_vv[i]
                    arr_s1[1, :, :] = s1_vh[i]
                    # arr_s1 = s1[i].reshape(count_s1, self.size, self.size)


                    arr_s1 = np.clip(arr_s1, -50, 1)
                    arr_s1 = (arr_s1 + 50) / 51

                    flood_data.append(arr_s1)


        elif self.choice == 2:
            s2,count_s2 = cut(self.s2_path, self.size)
            for i in range(0, len(s2)):
                arr_s2 = s2[i].reshape(count_s2, self.size, self.size)[1:, :, :]
                flood_data.append(arr_s2)

        else:
            flood_data = []
            if self.s1_path==self.s1vh_path:
                s1,count_s1 = cut(self.s1_path,self.size)
                s2, count_s2 = cut(self.s2_path, self.size)
                for i in range(0, len(s1)):
                    arr_s1 = s1[i].reshape(count_s1, self.size, self.size)
                    arr_s2 = s2[i].reshape(count_s2, self.size, self.size)[1:, :, :]
                    arr_s1 = np.clip(arr_s1, -50, 1)
                    arr_s1 = (arr_s1 + 50) / 51

                    flood_data.append((arr_s1,arr_s2))
            else:
                s1_vv, count_s1 = cut(self.s1_path, self.size)
                s1_vh, count_s1 = cut(self.s1vh_path, self.size)
                s2, count_s2 = cut(self.s2_path, self.size)
                for i in range(0, len(s1_vv)):
                    arr_s2 = s2[i].reshape(count_s2, self.size, self.size)[1:, :, :]
                    arr_s1 = np.zeros((2, 512, 512), dtype=float)
                    arr_s1[0, :, :] = s1_vv[i]
                    arr_s1[1, :, :] = s1_vh[i]
                    # arr_s1 = s1[i].reshape(count_s1, self.size, self.size)

                    arr_s1 = np.clip(arr_s1, -50, 1)
                    arr_s1 = (arr_s1 + 50) / 51

                    flood_data.append((arr_s1,arr_s2))



        logging.info('cut images finish')
        return flood_data






    """
    make multi-channel data
    """
nor = [
            [1612.2641794179053, 694.6404158569574],
            [1379.8899556061613, 734.589213934987],
            [1344.4295633683826, 731.6118897277566],
            [1195.157229000143, 860.603745394514],
            [1439.168369746529, 771.3569863637912],
            [2344.2498120705645, 921.6300590130161],
            [2796.473722876989, 1088.0256714514674],
            [2578.4108992777597, 1029.246558060433],
            [3023.817505678254, 1205.1064480965915],
            [476.7287418382585, 331.6878880293502],
            [59.24111403905757, 130.40242222578226],
            [1989.1945548720423, 993.7071664926801],
            [1152.4886461779677, 768.8907975412457],
            [-0.2938129263276281, 0.21578320121968173],
            [-0.36928344017880277, 0.19538602918264955],
            [-6393.00646782349, 0.19538602918264955],
            [-2398.566478742078, 0.19538602918264955]
        ]




def processTestIm_df(sen1,sen2):
        """
        make multi-channel data with Sentinel-1 bands, Sentinel-2 bands and water indices

                """
        s1,s2=sen1.copy(),sen2.copy()
        norm = transforms.Normalize([0.6851,
                                     0.5235,
                                     nor[0][0],
                                     nor[1][0],
                                     nor[2][0],
                                     nor[3][0],
                                     nor[4][0],
                                     nor[5][0],
                                     nor[6][0],
                                     nor[7][0],
                                     nor[8][0],
                                     nor[9][0],
                                     nor[10][0],
                                     nor[11][0],
                                     nor[12][0],
                                     nor[13][0],
                                     nor[14][0]

                                     ],
                                    [0.0820,
                                     0.1102,
                                     nor[0][1],
                                     nor[1][1],
                                     nor[2][1],
                                     nor[3][1],
                                     nor[4][1],
                                     nor[5][1],
                                     nor[6][1],
                                     nor[7][1],
                                     nor[8][1],
                                     nor[9][1],
                                     nor[10][1],
                                     nor[11][1],
                                     nor[12][1],
                                     nor[13][1],
                                     nor[14][1]]
                                    )

        # convert to PIL for easier transforms
        band_list = []
        for i in range(0, len(s1)):
            band = Image.fromarray(s2[i]).resize((512, 512))
            band_list.append(band)
        for i in range(0, len(s2)):
            band = Image.fromarray(s2[i]).resize((512, 512))
            band_list.append(band)

        ndw = (s2[2] - s2[7]) / (s2[2] + s2[7])
        ndw[np.isnan(ndw)] = 0
        ndwi = Image.fromarray(ndw).resize((512, 512))
        band_list.append(ndwi)

        ndm = (3.0 * s2[2] - s2[1] + 2.0 * s2[3] - 5.0 * s2[7]) / (3.0 * s2[2] + s2[1] + 2 * s2[3] + 5.0 * s2[7])
        ndm[np.isnan(ndm)] = 0
        ndmbwi = Image.fromarray(ndm).resize((512, 512))
        band_list.append(ndmbwi)

        bands_list = []
        for i in range(0, len(band_list)):
            bands = [F.crop(band_list[i], 0, 0, 256, 256), F.crop(band_list[i], 0, 256, 256, 256),
                     F.crop(band_list[i], 256, 0, 256, 256), F.crop(band_list[i], 256, 256, 256, 256)]
            bands_list.append(bands)

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze(),
                            transforms.ToTensor()(x3).squeeze(),
                            transforms.ToTensor()(x4).squeeze(),
                            transforms.ToTensor()(x5).squeeze(),
                            transforms.ToTensor()(x6).squeeze(),
                            transforms.ToTensor()(x7).squeeze(),
                            transforms.ToTensor()(x8).squeeze(),
                            transforms.ToTensor()(x9).squeeze(),
                            transforms.ToTensor()(x10).squeeze(),
                            transforms.ToTensor()(x11).squeeze(),
                            transforms.ToTensor()(x12).squeeze(),
                            transforms.ToTensor()(x13).squeeze(),
                            transforms.ToTensor()(x14).squeeze(),
                            transforms.ToTensor()(x15).squeeze(),
                            transforms.ToTensor()(x16).squeeze(),
                            transforms.ToTensor()(x17).squeeze()
                            ))
               for (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17) in
               zip(bands_list[0], bands_list[1], bands_list[2], bands_list[3], bands_list[4], bands_list[5],
                   bands_list[6],bands_list[7], bands_list[8], bands_list[9], bands_list[10], bands_list[11], bands_list[12],
                   bands_list[13], bands_list[14], bands_list[15], bands_list[16])]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)

        return ims

def processTestIm_s1(sen1):
        """
        make multi-channel data with Sentinel-1 bands
        """
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
        im = sen1.copy()


        # convert to PIL for easier transforms
        vv = Image.fromarray(im[0]).resize((512, 512))
        vh = Image.fromarray(im[1]).resize((512, 512))

        vvs = [F.crop(vv, 0, 0, 256, 256), F.crop(vv, 0, 256, 256, 256),
               F.crop(vv, 256, 0, 256, 256), F.crop(vv, 256, 256, 256, 256)]
        vhs = [F.crop(vh, 0, 0, 256, 256), F.crop(vh, 0, 256, 256, 256),
               F.crop(vh, 256, 0, 256, 256), F.crop(vh, 256, 256, 256, 256)]

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze()))
               for (x1, x2) in zip(vvs, vhs)]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)



        return ims

def processTestIm_band12(sen1,sen2):
        """
                make multi-channel data with Sentinel-1 bands and Band12 from Sentinel-2
                """
        s1,s2=sen1.copy(),sen2.copy()
        norm = transforms.Normalize([0.6851, 0.5235, nor[12][0]], [0.0820, 0.1102, nor[12][1]])

        # convert to PIL for easier transforms
        vv = Image.fromarray(s1[0]).resize((512, 512))
        vh = Image.fromarray(s1[1]).resize((512, 512))
        band12 = Image.fromarray(s2[12]).resize((512, 512))

        vvs = [F.crop(vv, 0, 0, 256, 256), F.crop(vv, 0, 256, 256, 256),
               F.crop(vv, 256, 0, 256, 256), F.crop(vv, 256, 256, 256, 256)]
        vhs = [F.crop(vh, 0, 0, 256, 256), F.crop(vh, 0, 256, 256, 256),
               F.crop(vh, 256, 0, 256, 256), F.crop(vh, 256, 256, 256, 256)]
        band12s = [F.crop(band12, 0, 0, 256, 256), F.crop(band12, 0, 256, 256, 256),
                   F.crop(band12, 256, 0, 256, 256), F.crop(band12, 256, 256, 256, 256)]

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze(),
                            transforms.ToTensor()(x3).squeeze()))
               for (x1, x2, x3) in zip(vvs, vhs, band12s)]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)

        return ims



def processTestIm_s2(sen2):
        """
        make multi-channel data with Sentinel-2 bands
        """
        s2=sen2.copy()
        norm = transforms.Normalize([
                                     nor[0][0],
                                     nor[1][0],
                                     nor[2][0],
                                     nor[3][0],
                                     nor[4][0],
                                     nor[5][0],
                                     nor[6][0],
                                     nor[7][0],
                                     nor[8][0],
                                     nor[9][0],
                                     nor[10][0],
                                     nor[11][0],
                                     nor[12][0]

                                     ],
                                    [
                                     nor[0][1],
                                     nor[1][1],
                                     nor[2][1],
                                     nor[3][1],
                                     nor[4][1],
                                     nor[5][1],
                                     nor[6][1],
                                     nor[7][1],
                                     nor[8][1],
                                     nor[9][1],
                                     nor[10][1],
                                     nor[11][1],
                                     nor[12][1]]
                                    )

        # convert to PIL for easier transforms
        band_list = []
        for i in range(0, len(s2)):
            band = Image.fromarray(s2[i]).resize((512, 512))
            band_list.append(band)


        bands_list = []
        for i in range(0, len(band_list)):
            bands = [F.crop(band_list[i], 0, 0, 256, 256), F.crop(band_list[i], 0, 256, 256, 256),
                     F.crop(band_list[i], 256, 0, 256, 256), F.crop(band_list[i], 256, 256, 256, 256)]
            bands_list.append(bands)

        ims = [torch.stack((transforms.ToTensor()(x1).squeeze(),
                            transforms.ToTensor()(x2).squeeze(),
                            transforms.ToTensor()(x3).squeeze(),
                            transforms.ToTensor()(x4).squeeze(),
                            transforms.ToTensor()(x5).squeeze(),
                            transforms.ToTensor()(x6).squeeze(),
                            transforms.ToTensor()(x7).squeeze(),
                            transforms.ToTensor()(x8).squeeze(),
                            transforms.ToTensor()(x9).squeeze(),
                            transforms.ToTensor()(x10).squeeze(),
                            transforms.ToTensor()(x11).squeeze(),
                            transforms.ToTensor()(x12).squeeze(),
                            transforms.ToTensor()(x13).squeeze()
                            ))
               for (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,x13) in
               zip(bands_list[0], bands_list[1], bands_list[2], bands_list[3], bands_list[4], bands_list[5],
                   bands_list[6],bands_list[7], bands_list[8], bands_list[9], bands_list[10], bands_list[11], bands_list[12]
                   )]

        ims = [norm(im) for im in ims]
        ims = torch.stack(ims)

        return ims








