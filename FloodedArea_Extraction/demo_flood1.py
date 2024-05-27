# -*- coding: utf-8 -*-
# @Author  : Miao Shen
# @Time    : 2022/8/21
# @File    : flood_extraction_demo.py



import argparse
from model.flood import FloodInundation
from preprocessing.dataprocess import CutImage,flood_extraction_out
from  nn.Unetplusplus import NestedUNet,VGGBlock


def get_args():
    """
    Example

    ------------
    .. code-block:: python
            import numpy
            # this part can be replaced by certain demo

    """


    parse = argparse.ArgumentParser(description="flood inundation extraction",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument("-img1", "--sentinel1_pre", dest="sentinel1_pre",required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image\Lao_s1_pre.tif", help="sentinel-1 image before disaster")
    parse.add_argument("-img1vh", "--sentinel1vh_pre", dest="sentinel1vh_pre", required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image\Lao_s1_pre.tif",
                       help="sentinel-1 vh image before disaster")
    parse.add_argument("-img2", "--sentinel2_pre", dest="sentinel2_pre",required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image\Lao_s2_pre.tif", help="sentinel-2 image before disaster")
    parse.add_argument("-img3", "--sentinel1_post", dest="sentinel1_post",required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image\Lao_s1_post.tif", help="sentinel-1 image after disaster")
    parse.add_argument("-img3vh", "--sentinel1vh_post", dest="sentinel1vh_post", required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image\Lao_s1_post.tif",
                       help="sentinel-1 vh image after disaster")
    parse.add_argument("-img4", "--sentinel2_post", dest="sentinel2_post",required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image\Lao_s2_post.tif", help="sentinel-2 image after disaster")

    parse.add_argument("-k1", "--threshold1", dest="threshold1", type=float, default=0.4,
                       help="the first threshold for decision-level data fusion")
    parse.add_argument("-k2", "--threshold2", dest="threshold2", type=float, default=0.95,
                       help="the second threshold for decision-level data fusion")
    parse.add_argument("-k3", "--threshold3", dest="threshold3", type=float, default=0.5,
                       help="the third threshold for decision-level data fusion")

    parse.add_argument("-c", "--choice", dest="choice",
                       type=int, default=3, help="select Sentienl-1 or Sentinel-2 or the fusion of Sentinel-1 and Sentinel-2 to extract flood inundation areas1: only use Sentienl-1; 2: only use Sentienl-2; 3: use data fusion")

    parse.add_argument("-o", "--output", dest="output", required=True,
                       type=str, default="D:\BaiduNetdiskDownload\image/output.tif", help="output file path")


    return parse.parse_args()


if __name__ == '__main__':

    args=get_args()
    print("load data succeed")

    s1_pre_path = args.sentinel1_pre
    s1vh_pre_path= args.sentinel1vh_pre
    s2_pre_path = args.sentinel2_pre
    s1_post_path = args.sentinel1_post
    s1vh_post_path = args.sentinel1vh_post
    s2_post_path = args.sentinel2_post

    k1 = args.threshold1
    k2 = args.threshold2
    k3 = args.threshold3

    choice = args.choice
    output = args.output




    # cut the iamge into 512 pixels * 512 pixels
    CutImage_pre=CutImage(s1_path=s1_pre_path,s1vh_path=s1vh_pre_path,s2_path=s2_pre_path,choice=choice)
    cutimage_pre=CutImage_pre.run()
    print("pre_image cut finish")
    CutImage_post = CutImage(s1_path=s1_post_path, s1vh_path=s1vh_post_path,s2_path=s2_post_path,choice=choice)
    cutimage_post=CutImage_post.run()
    print("post_image cut finish")

    profile=CutImage_pre.getprofile()

    # flood inundation extraction
    model = FloodInundation(data_pre=cutimage_pre,data_post=cutimage_post,k1=k1,k2=k2,k3=k3,output=output,profile=profile,choice=choice)
    print("flood inundation extraction method construct finished.")
    res=model.extraction()
    print("ready to output.")
    flood_extraction_out(profile,res,output)
    print("flood inundation extraction finished.")
