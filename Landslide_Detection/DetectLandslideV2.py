import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Polygon
from skimage import io
# Import Mask RCNN
from mrcnn import visualize
from mrcnn.visualize import find_contours
import mrcnn.model as modellib

from samples.landslide import landslide_v2 as landslide


############################################################
#  Basic
############################################################
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


############################################################
#  Visualize
############################################################
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, alpha=0.3, file_path=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    # colors = colors or visualize.random_colors(N)
    colors = colors or visualize.random_colors(1)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # color = colors[i]
        color = colors[0]  # 都使用红色

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color, alpha)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.margins(0, 0)
        if file_path:
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0.0)
            #             plt.savefig(file_path,bbox_inches='tight',dpi=300,pad_inches=0.0)
            print('saved {}'.format(file_path))
            plt.close('all')
        # plt.show()


def save_masks(r, result_path):
    masks = r['masks']

    temp = masks.copy()
    temp = np.array(temp, dtype=np.uint8)
    temp[temp == 1] = 255

    h, w, c = temp.shape[:3]
    print(h, w, c)

    if temp.shape[2] == 0:
        io.imsave(result_path, np.zeros((h, w, 1), dtype=np.uint8))
    elif temp.shape[2] == 1:
        io.imsave(result_path, temp)
    else:
        temp = np.amax(temp, axis=2)  # merge
        io.imsave(result_path, temp)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect landslides.')
    parser.add_argument('--image', required=True,
                        metavar="/path/to/landslide/image/",
                        help='Path of the satellite image')
    parser.add_argument('--dem', required=True,
                        metavar="/path/to/landslide/image/",
                        help='Path of the satellite image')
    parser.add_argument('--slope', required=True,
                        metavar="/path/to/landslide/image/",
                        help='Path of the satellite image')
    parser.add_argument('--river', required=True,
                        metavar="/path/to/landslide/image/",
                        help='Path of the satellite image')
    parser.add_argument('--result', required=True,
                        metavar="/path/to/extraction/result",
                        help="Path to result")
    args = parser.parse_args()

    image_path = args.image
    # additional attributes
    dem_path = args.dem
    slope_path = args.slope
    water_path = args.river
    result_path = args.result

    config = landslide.LandslideConfig()


    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    config.display()

    # GPU for training.
    DEVICE = "/gpu:1"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"

    LANDSLIDE_DIR = "../images/landslides"
    # Load validation dataset
    dataset = landslide.LandslideDataset()
    dataset.load_data_for_detection(image_path, dem_path, slope_path, water_path)
    # Must call before using the dataset
    dataset.prepare()
    print("Validation Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)

    weights_path = "model/mask_rcnn_landslide_0032.h5"
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    ## save prediction images
    result_image_folder = os.path.join(result_path, 'extraction_result')
    if not os.path.exists(result_image_folder):
        os.makedirs(result_image_folder)

    result_mask_folder = os.path.join(result_path, 'extraction_mask')
    if not os.path.exists(result_mask_folder):
        os.makedirs(result_mask_folder)

    image_ids = dataset.image_ids
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]

        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        results = model.detect([image], verbose=1)
        r = results[0]

        ## image
        result_image_path = os.path.join(result_image_folder, info["id"])
        display_instances(image[:, :, :3], r['rois'], r['masks'], r['class_ids'],
                          dataset.class_names, r['scores'], alpha=0.3, file_path=result_image_path)

        # save prediction masks
        result_mask_path = os.path.join(result_mask_folder, info["id"])
        save_masks(r, result_mask_path)
