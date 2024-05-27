import os
import json
import argparse
import xml.etree.cElementTree as ET
import cv2
import random
from PIL import Image, ImageDraw, ImageFont

FONT_SIZE = 13 * 2
#WIDTH = 2
#COLOR_LIST = ["red", "green", "blue", "purple"]
'''
COLOR_LIST = ["red", "green", "blue", "cyan", "yellow", "purple",
              "deeppink", "ghostwhite", "darkcyan", "olive",
              "orange", "orangered", "darkgreen"]
'''


class ShowResult(object):
    def __init__(self, args):
        self.args = args
        self.annotation_path = self.args.annotation_path
        self.save_picture_path = self.args.save_picture_path
        self.origin_img_path = self.args.origin_img_path

        # 保存原有类别
        self.category = {'collapsed': 'collapsed'}

    def category_id_convert_to_name(self, category):
        return self.category[category]

    def GetAnnotBoxLoc(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        ObjBndBoxSet = {}
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(float(BndBox.find('xmin').text))
            y1 = int(float(BndBox.find('ymin').text))
            x2 = int(float(BndBox.find('xmax').text))
            y2 = int(float(BndBox.find('ymax').text))
            BndBoxLoc = [x1, y1, x2, y2]
            if ObjName in ObjBndBoxSet:
                ObjBndBoxSet[ObjName].append(BndBoxLoc)
            else:
                ObjBndBoxSet[ObjName] = [BndBoxLoc]
        return ObjBndBoxSet

    def GetAnnotName(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        path = root.find('path').text
        return path

    def Drawpic(self):
        if not os.path.exists(self.save_picture_path):
            os.mkdir(self.save_picture_path)

        xml_list = os.listdir(self.annotation_path)
        # 读取每一个xml文件
        for idx, xml in enumerate(xml_list):
            xml_file = os.path.join(self.annotation_path, xml)

            # box是一个字典，以类别为key，box为值
            box = self.GetAnnotBoxLoc(xml_file)
            img_name = str(xml).replace(".xml", ".jpg")

            img_path = os.path.join(self.origin_img_path, img_name)
            print("当前正在处理第-{0}-张图片, 总共需要处理-{1}-张, 完成百分比:{2:.2%}".format(idx + 1,
                                                                        len(xml_list),
                                                                        (idx + 1) / len(xml_list)))
            # 对每一个bbox标注
            img = Image.open(img_path, "r")  # img1.size返回的宽度和高度(像素表示)
            draw = ImageDraw.Draw(img)

            for classes in list(box.keys()):
                #COLOR = random.choice(COLOR_LIST)
                category_name = self.category_id_convert_to_name(classes)
                for boxes in box[classes]:
                    x_left = int(boxes[0])
                    y_top = int(boxes[1])
                    x_right = int(boxes[2])
                    y_down = int(boxes[3])

                    top_left = (int(boxes[0]), int(boxes[1]))  # x1,y1
                    top_right = (int(boxes[2]), int(boxes[1]))  # x2,y1
                    down_right = (int(boxes[2]), int(boxes[3]))  # x2,y2
                    down_left = (int(boxes[0]), int(boxes[3]))  # x1,y2

                    draw.line([top_left, top_right, down_right, down_left, top_left])
                    draw.text((x_left + 30, y_top - FONT_SIZE), str(category_name))
            # 存储图片
            img_name = str(img_name).replace('.jpg', '.png')
            save_path = os.path.join(self.save_picture_path, img_name)

            img.save(save_path, "png")


def main():
    parser = argparse.ArgumentParser()
    # annotation_path
    parser.add_argument('-annotation_path', default=r"VOCdevkit/VOC2007/Annotations",
                        help='the single img json file path')
    # save_picture_path
    parser.add_argument('-save_picture_path', default=r"VOCdevkit/VOC2007/annotation_image",
                        help='the val img result json file path')
    # origin_img_path
    parser.add_argument('-origin_img_path', default=r'VOCdevkit/VOC2007/JPEGImages',
                        help='the val img path root')

    args = parser.parse_args()
    showresult = ShowResult(args)

    showresult.Drawpic()


if __name__ == '__main__':
    main()