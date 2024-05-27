import os

import argparse

parser = argparse.ArgumentParser(description='Run Building Damage Model')
parser.add_argument('--ImgDir_input', type=str, default="./data/images/", help='the image input directory')

args = parser.parse_args()

inputDir = args.ImgDir_input

os.system("python predict_cls.py --ImgDir_input {}".format(inputDir))
