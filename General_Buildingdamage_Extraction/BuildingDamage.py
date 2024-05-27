import os

import argparse

parser = argparse.ArgumentParser(description='Run Building Damage Model')
parser.add_argument('--ImgDir_input', type=str, default="./data/images/", help='the image input directory')
parser.add_argument('--PredictDir_output', type=str, default="./data/predict/", help='the prediction output directory')

args = parser.parse_args()

inputDir = args.ImgDir_input
outputDir = args.PredictDir_output

os.system("python predict50_loc.py --ImgDir_input {}".format(inputDir))
os.system("python predict50cls.py --seed 0 --ImgDir_input {}".format(inputDir))
os.system("python predict50cls.py --seed 1 --ImgDir_input {}".format(inputDir))
os.system("python predict50cls.py --seed 2 --ImgDir_input {}".format(inputDir))
os.system("python create_submission50.py --ImgDir_input {} --PredictDir_output {}".format(inputDir, outputDir))
