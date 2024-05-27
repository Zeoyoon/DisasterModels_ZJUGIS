"""
Mask R-CNN
Train on the toy landslide dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 landslide.py train --dataset=/path/to/landslide/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 landslide.py train --dataset=/path/to/landslide/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 landslide.py train --dataset=/path/to/landslide/dataset --weights=imagenet

    # Apply color splash to an image
    python3 landslide.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 landslide.py splash --weights=last --video=<URL or path to file>
"""

import datetime
import os
import sys
import json

import numpy as np
import skimage.draw
from tifffile import imread

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

SAMPLES_DIR = '/home/yrl/prepare/ludian/samples'


############################################################
#  Configurations
############################################################


class LandslideConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "landslide"

    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + landslide

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    EPOCHS = 20

    VALIDATION_STEPS = 300

    IMAGE_CHANNEL_COUNT = 6

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([99.146, 112.589, 112.923, 0.0, 0.0, 0.0])
    MEAN_PIXEL = np.array([99.146, 112.589, 112.923, 0.0, 0.0, 0.0])

    # slope 95.565
    # slope 95.56493123129731
    # dem 106.78596247201942

    LEARNING_RATE = 0.001

    WEIGHT_DECAY = 0.005


############################################################
#  Dataset
############################################################

class LandslideDataset(utils.Dataset):

    def load_data_for_detection(self, data_dir, dem_dir, slope_dir, water_dir):
        """Load a subset of the Landslide dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("landslide", 1, "landslide")

        folder = os.path.dirname(data_dir)
        file = data_dir.split('/')[-1]
        image = skimage.io.imread(data_dir)
        height, width = image.shape[:2]

        self.add_image(
            "landslide",
            image_id=file,  # use file name as a unique image id
            path=data_dir,
            water=water_dir,
            dem=dem_dir,
            slope=slope_dir,
            width=width, height=height,
            polygons=[])

    def load_landslide(self, dataset_dir):
        """Load a subset of the Landslide dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("landslide", 1, "landslide")

        # Train or validation dataset?
        # assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir)

        # Load samples
        image_list = os.listdir(os.path.join(dataset_dir))

        for file in image_list:
            image_name = file.split('.')[0]
            image_path = os.path.join(dataset_dir, file)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # load annotations
            a = json.load(open(os.path.join(SAMPLES_DIR, 'annotation', image_name + '.json')))
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            self.add_image(
                "landslide",
                image_id=file,  # use file name as a unique image id
                path=image_path,
                water=os.path.join(SAMPLES_DIR, 'water', file),
                dem=os.path.join(SAMPLES_DIR, 'dem', file),
                slope=os.path.join(SAMPLES_DIR, 'slope', file),
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a landslide dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "landslide":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "landslide":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def add_image(self, source, image_id, path, water, dem, slope, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            "dem": dem,
            "water": water,
            "slope": slope
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # Load landslide-inducing data
        water = skimage.io.imread(self.image_info[image_id]['water'])
        dem = skimage.io.imread(self.image_info[image_id]['dem'])
        slope = skimage.io.imread(self.image_info[image_id]['slope'])

        # for tif files
        # water = imread(self.image_info[image_id]['water'])
        # dem = imread(self.image_info[image_id]['dem'])
        # slope = imread(self.image_info[image_id]['slope'])

        # Load original image
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Load landslide-inducing data
        dem = dem[:, :, np.newaxis]
        water = water[:, :, np.newaxis]
        slope = slope[:, :, np.newaxis]
        fake = np.zeros((image.shape[0], image.shape[1], 1))

        # merge
        result = np.concatenate((image, dem, slope, water), axis=2)

        return result


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LandslideDataset()
    dataset_train.load_landslide(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LandslideDataset()
    dataset_val.load_landslide(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.EPOCHS,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Validation
############################################################
def compute_precision(gt_boxes, gt_class_ids, gt_masks,
                      pred_boxes, pred_class_ids, pred_scores, pred_masks,
                      iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    tp: correct prediction among landslide samples
    fn: missed prediction among landslide samples
    fp: incorrect prediction among non-landslide samples
    tn: correct prediction among non-landslide samples
    """

    # non-landslide sample
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return 0, 0, 0, 1, 0
        else:
            return 0, 0, 1, 0, 0

    # lanslide sample
    else:
        # Get matches and overlaps
        gt_match, pred_match, overlaps = utils.compute_matches(
            gt_boxes, gt_class_ids, gt_masks,
            pred_boxes, pred_class_ids, pred_scores, pred_masks,
            iou_threshold)

        # Compute precision and recall at each prediction box step
        # pred_match中数组中值为-1，表示该预测没有对应的真实值
        tp = np.sum(pred_match > -1)
        # fp = len(pred_match) - tp
        fn = len(gt_match) - np.sum(gt_match > -1)
        return tp, fn, 0, 0, overlaps


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect landslides.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/landslide/dataset/",
                        help='Directory of the landslide dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LandslideConfig()
    else:
        class InferenceConfig(LandslideConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "conv1"])

    elif args.weights.lower() == "none":
        print(1)
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
