# -*- coding: utf-8 -*-
# @Time : 2022/8/21
# @Author : Miao Shen
# @File : Flood Inundation Extraction

# from pydde.core.model import BaseProcedureModel
import numpy as np
import torch
import torch.nn as nn
import rasterio
from preprocessing.dataprocess import CutImage
import sys
sys.path.append("..")
from preprocessing.dataprocess import processTestIm_df,processTestIm_s1,processTestIm_band12,processTestIm_s2
import preprocessing.dataprocess
from  nn.Unetplusplus import NestedUNet,VGGBlock





class FloodInundation():
    """
    FloodInundation is used to achieve a decision-level data fusion method. The data fusion
    method combines three different UNet++ models trained by Sentinel-1 and Sentinel-2 data,
    which produces more accurate flood inundated areas.

    """


    def __init__(self, data_pre, data_post, k1, k2, k3,output,profile,choice,size=512):
        """

        :param data_pre: numpy array
            multi-channel data before disaster.
        :param data_post: numpy array
            multi-channel data before disaster.
        :param choice: int, default=3
            If choice=1, the method will extract flood inundated areas only by Sentinel-1 images.If
            choice=2,the method will extract flood inundated areas only by Sentinel-2 images. If
            choice=3,the method will fuse Sentinel-1 and Sentinel-2 images by decision-level data
            fusion method.
        :param k1: float, default=0.4
            The first threshold for decision-level data fusion method. In the first step of
            the decision-level data fusion method, the model good at removing non-water areas
            is used to calculate the probability of each pixel classified as water. The pixels
            with lower than k1 is defined as non-water.
        :param k2: float, default=0.95
            The second threshold for decision-level data fusion method. In the second step of
            the decision-level data fusion method, the model good at distinguish water in cloud-free
            area is used to calculate the probability of each pixel classified as water. The pixels
            with higher than k2 is defined as water.
        :param k3: float, default=0.5
            The third threshold for decision-level data fusion method. In the third step of
            the decision-level data fusion method, the model good at distinguish water in cloud
            area is used to extract flood inundation areas.
        :param output: str
            path of output image
        :param image_path: str
            path of original image(use to get profile)
        :param size: int, default=512
        :param choice: int, default=3

        """

        super().__init__()

        self.data_pre = data_pre
        self.data_post = data_post
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.output=output
        self.profile = profile
        self.height = profile["height"]
        self.width = profile["width"]
        self.size=size
        self.choice=choice


        self.model_s1 = torch.load("./data/checkpoint/Peru_s1.cp")
        self.model_band12 = torch.load("./data/checkpoint/Peru_s1+band12.cp")
        self.model_df = torch.load("./data/checkpoint/Peru_df.cp")
        self.model_s2 = torch.load("./data/checkpoint/Peru_s2.cp")

    def extraction(self):
        """
        extract flood inundated areas
        :return: results of flood inundated areas
            results of flood inundated areas in size of 512*512
        """


        img = []

        with torch.no_grad():
            data_pre=self.data_pre
            data_post=self.data_post

            for i in range(0, len(data_post)):
                if self.choice == 3:
                    #before disaster
                    s1_pre,s2_pre=data_pre[i]
                    images_pre_band12 = processTestIm_band12(s1_pre,s2_pre)
                    images_pre_s1 = processTestIm_s1(s1_pre)
                    images_pre_df = processTestIm_df(s1_pre,s2_pre)
                    images_pre = (images_pre_s1, images_pre_band12, images_pre_df)

                    # after disaster
                    s1_post, s2_post = data_post[i]
                    images_post_band12 = processTestIm_band12(s1_post, s2_post )
                    images_post_s1 = processTestIm_s1(s1_post)
                    images_post_df = processTestIm_df(s1_post, s2_post )
                    images_post=(images_post_s1,images_post_band12,images_post_df)

                    outputs_pre = self.data_fusion(images_pre)
                    outputs_post = self.data_fusion(images_post)

                elif self.choice==2:
                    images_pre_s2 = processTestIm_s2(data_pre[i])
                    images_post_s2 = processTestIm_s2(data_post[i])
                    outputs_pre = self.data_fusion_s2(images_pre_s2)
                    outputs_post = self.data_fusion_s2(images_post_s2)

                else:

                    images_pre_s1 = processTestIm_s1(data_pre[i])
                    images_post_s1 = processTestIm_s1(data_post[i])
                    outputs_pre = self.data_fusion_s1(images_pre_s1)
                    outputs_post = self.data_fusion_s1(images_post_s1)

                outputs_pre = outputs_pre.flatten()
                outputs_post = outputs_post.flatten()



                output = torch.sub(outputs_post, outputs_pre)
                # output = outputs_pre
                zero_out = torch.zeros_like(output)
                output = torch.where(output < 0, zero_out, output)

                output = output.cpu().numpy()
                output = np.array(output)
                output = output.reshape((4, 256, 256))

                temp_out = np.zeros(shape=(512, 512), dtype=np.uint8)
                temp_out[0:256, 0:256] = output[0]
                temp_out[256:512, 0:256] = output[2]
                temp_out[0:256, 256:512] = output[1]
                temp_out[256:512, 256:512] = output[3]


                img.append(temp_out)

        return img

    def extraction_single(self):
        """
        extract flood inundated areas
        :return: results of water areas
            results of water areas in size of 512*512
        """


        img = []

        with torch.no_grad():

            data_post=self.data_post

            for i in range(0, len(data_post)):
                if self.choice == 3:
                    # after disaster
                    s1_post, s2_post = data_post[i]
                    images_post_band12 = processTestIm_band12(s1_post, s2_post )
                    images_post_s1 = processTestIm_s1(s1_post)
                    images_post_df = processTestIm_df(s1_post, s2_post )
                    images_post=(images_post_s1,images_post_band12,images_post_df)

                    outputs_post = self.data_fusion(images_post)

                elif self.choice==2:
                    images_post_s2 = processTestIm_s2(data_post[i])
                    outputs_post = self.data_fusion_s2(images_post_s2)

                else:

                    images_post_s1 = processTestIm_s1(data_post[i])
                    outputs_post = self.data_fusion_s1(images_post_s1)


                outputs_post = outputs_post.flatten()



                output = outputs_post
                # output = outputs_pre
                zero_out = torch.zeros_like(output)
                output = torch.where(output < 0, zero_out, output)

                output = output.cpu().numpy()
                output = np.array(output)
                output = output.reshape((4, 256, 256))

                temp_out = np.zeros(shape=(512, 512), dtype=np.uint8)
                temp_out[0:256, 0:256] = output[0]
                temp_out[256:512, 0:256] = output[2]
                temp_out[0:256, 256:512] = output[1]
                temp_out[256:512, 256:512] = output[3]


                img.append(temp_out)

        return img



    def data_fusion_s1(self,test_data):
        """
        data fusion method for Sentinel-1
        :param test_data: numpy array
           multi-channel data
        :return: numpy array
            result extracted by Sentinel-1 data
        """

        images_s1 = test_data

        model_s1 = self.model_s1.eval()
        model_s1 = model_s1.cuda()
        # print(images_s1[0])
        # print("_____________")

        out_s1 = model_s1(images_s1.cuda())

        pro_s1 = nn.functional.softmax(out_s1, dim=1)
        pro_s1 = pro_s1[:, 1, :, :]




        # fusion
        outputs_1 = torch.where(pro_s1 >= 0.5, 1, 0)




        return outputs_1


    def data_fusion_s2(self,test_data):
        """
        data fusion method for Sentinel-1
        :param test_data: numpy array
           multi-channel data
        :return: numpy array
            result extracted by Sentinel-1 data
        """

        images_s1 = test_data

        model_s2 = self.model_s2.eval()
        model_s2 = model_s2.cuda()

        out_s2 = model_s2(images_s1.cuda())

        pro_s2 = nn.functional.softmax(out_s2, dim=1)
        pro_s2 = pro_s2[:, 1, :, :]

        # fusion
        outputs_1 = torch.where(pro_s2 >= 0.5, 1, 0)

        return outputs_1

    def data_fusion(self,test_data):
        """
        decision-level data fusion method for combining  Sentinel-1 and Sentinel-2

        step 1 : remove non-water

        step 2 : extract water in non-cloud area

        step 3 : extract water in cloud area

        :param test_data: numpy array
            multi-channel data
        :return: numpy array
            result extracted by Sentinel-1 and Sentinel-2 data
        """


        model_s1 = self.model_s1.eval()
        model_s1 = model_s1.cuda()
        model_band12 = self.model_band12.eval()
        model_band12 = model_band12.cuda()
        model_df = self.model_df.eval()
        model_df = model_df.cuda()

        (images_s1,images_band12,images_df)=test_data

        out_band12 = model_band12(images_band12.cuda())
        out_s1 = model_s1(images_s1.cuda())
        out_df = model_df(images_df.cuda())


        pro_band12 = nn.functional.softmax(out_band12, dim=1)
        pro_band12 = pro_band12[:, 1, :, :]

        pro_s1 = nn.functional.softmax(out_s1, dim=1)
        pro_s1 = pro_s1[:, 1, :, :]
        pro_df = nn.functional.softmax(out_df, dim=1)
        pro_df = pro_df[:, 1, :, :]



        # step 1 : remove non-water
        outputs_1 = torch.add(pro_band12 * 0.5, pro_s1 * 0.5)
        outputs_1[outputs_1 < 0.1] = 0
        outputs_1[outputs_1 >= 0.1] = 2

        # step 2 : extract water in non-cloud area
        outputs_2 = torch.zeros_like(pro_band12)
        outputs_2[pro_df < 0.02] = 0
        outputs_2[pro_df >= 0.02] = 2
        outputs_2[pro_df >= 0.95] = 1

        # step 3 : extract water in cloud area
        outputs_3 = torch.add(pro_df * 0.5, pro_s1 * 0.5)
        outputs_3[outputs_3 >= 0.5] = 1
        outputs_3[outputs_3 < 0.5] = 0

        # fusion
        outputs_1 = torch.where(outputs_1 >= 2, outputs_2, outputs_1)
        outputs_1 = torch.where(outputs_1 >= 2, outputs_3, outputs_1)

        return outputs_1








