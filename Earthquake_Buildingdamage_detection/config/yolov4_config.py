# coding=utf-8
# project
DATA_PATH = r"/exstorage/slf007/3-YOLO_done/3-YOLO/data"
PROJECT_PATH = "/exstorage/slf007/3-YOLO_done/3-YOLO/data"
DETECTION_PATH = ""

MODEL_TYPE = {
    "TYPE": "YOLOv4"
}  # YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4

CONV_TYPE = {"TYPE": "GENERAL"}  # conv type:DO_CONV or GENERAL

ATTENTION = {"TYPE": "NONE"}  # 

# train
TRAIN = {
    "DATA_TYPE": "Customer",  # DATA_TYPE: VOC ,COCO or Customer
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 4,
    "MULTI_SCALE_TRAIN": False,
    "IOU_THRESHOLD_LOSS": 0.5,
    "YOLO_EPOCHS": 800,
    "Mobilenet_YOLO_EPOCHS": 960,
    "NUMBER_WORKERS": 0,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2,  # or None
    "showatt": False
}


# val
VAL = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 12,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.62,
    "NMS_THRESH": 0.45,
    "MULTI_SCALE_VAL": True,
    "FLIP_VAL": False,
    "Visual": True,
    #"showatt": False
}

Customer_DATA = {
    "NUM": 1,  # your dataset number
    "CLASSES": ["collapsed"],  # your dataset class
}

VOC_DATA = {
    "NUM": 1,
    "CLASSES": [
        "collapsed"
   ],
}

COCO_DATA = {
    "NUM": 1,
    "CLASSES": [
        "collapsed"
    ],
}


# model
MODEL = {
    "ANCHORS": [
        [
            (32, 30),
            (51, 69),
            (38, 46),
        ],  # Anchors for small obj(12,16),(19,36),(40,28)
        [
            (45, 80),
            (58, 39),
            (62, 59),
        ],  # Anchors for medium obj(36,75),(76,55),(72,146)
        [
            (79, 88),
            (58, 39),
            (62, 59)
        ],
    ],  # Anchors for big obj(142,110),(192,243),(459,401)

    #[35.84, 107.63], [69.02, 124.49], [94.17, 163.43], [184.33, 177.78], [300.75, 627.02], [442.42, 266.55], [442.76, 441.56], [481.95, 576.25], [503.37, 501.36]
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
}
