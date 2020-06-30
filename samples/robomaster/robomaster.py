#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


class RobomasterConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "robomaster"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + .....

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # IMAGE_MIN_DIM = 1080
    # IMAGE_MAX_DIM = 1920

    GPU_COUNT = 2
    # IMAGES_PER_GPU = 8
config = RobomasterConfig()
config.display()


# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# ## Dataset
#
# Create a synthetic dataset
#
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
#
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:

ROBOMASTER_DIR = os.path.join(ROOT_DIR, "robomaster")


def getImageIds(dir):
    imageIds = []
    file = open(ROBOMASTER_DIR + '/' + dir)
    try:
        for line in file:
            imageIds.append(line.strip('\n'))
    finally:
        file.close()
    return imageIds


import xml.etree.ElementTree as ET


def test_remove(xmin, ymin, xmax, ymax):
    return 0


class RobomasterDataset(utils.Dataset):

    def load_robomaster(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("robomaster", 1, "car")
        self.add_class("robomaster", 2, "watcher")
        self.add_class("robomaster", 3, "base")
        self.add_class("robomaster", 4, "ignore")
        self.add_class("robomaster", 5, "armor")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        imageIds = []
        if subset == 'train':
            imageIds = getImageIds('train.txt')
        elif subset == 'val':
            imageIds = getImageIds('val.txt')
        elif subset == 'test':
            imageIds = getImageIds('test.txt')

        assert dataset_dir in ["image_center", "image_zs", "image_ys", "image_zx", "image_yx"]
        # Load annotations
        # 我们对图像上场地内的所有机器人、哨兵机器人和基地包括它们身上对应的所有装甲板都有标注。

        # xml annotation文件解释：
        # filename：对应图片的名字，内容是英文，可能和文件名不一致但并不影响使用。
        # size：图像的width和height
        # object的属性有：
        # name：有5种属性值：car(机器人)，watcher（哨兵机器人），base（基地），ignore（忽略）, armor（装甲板）
        # bendbox：表示每个目标物体的boundingbox坐标值。左上角坐标：(xmin，ymin)，右下角坐标：（xmax，ymax）, 所有坐标值都是基于1920x1080的原始图像而言

        # Add images
        for image_id in imageIds:
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(ROBOMASTER_DIR, dataset_dir + '/' + image_id + '.jpg')
            # image = skimage.io.imread(image_path)
            # height, width = image.shape[:2]

            tree = ET.parse(ROBOMASTER_DIR + '/image_annotation/' + image_id + '.xml')
            root = tree.getroot()

            objects = [];
            for object in root.findall("object"):
                objects.append(object)

            self.add_image(
                "robomaster",
                image_id=image_id,  # use file name as a unique image id
                path=image_path,
                width=960,
                height=540,
                objects=objects,
                imagelocation=dataset_dir)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "robomaster":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        temp_objects = [] + info["objects"]
        imagelocation = info["imagelocation"]
        for i, object in enumerate(info["objects"]):
            xmin = int(float(object.find('bndbox').find('xmin').text))
            ymin = int(float(object.find('bndbox').find('ymin').text))
            xmax = int(float(object.find('bndbox').find('xmax').text))
            ymax = int(float(object.find('bndbox').find('ymax').text))

            location_xmin = 0
            location_ymin = 0
            location_xmax = 1920
            location_ymax = 1080
            if imagelocation == 'image_center':
                # [270:270+540, 480:480+960]
                location_xmin = 480
                location_ymin = 270
                location_xmax = 480 + 960
                location_ymax = 270 + 540
            if imagelocation == 'image_zs':
                # [0:540, 0:960]
                location_xmin = 0
                location_ymin = 0
                location_xmax = 960
                location_ymax = 540

            if imagelocation == 'image_ys':
                # [0:540, 960:1920]
                location_xmin = 960
                location_ymin = 0
                location_xmax = 1920
                location_ymax = 540

            if imagelocation == 'image_zx':
                # 【540:1080, 0:960]
                location_xmin = 0
                location_ymin = 540
                location_xmax = 960
                location_ymax = 1080

            if imagelocation == 'image_yx':
                # [540:1080, 960:1920]
                location_xmin = 960
                location_ymin = 540
                location_xmax = 1920
                location_ymax = 1080

            if xmin > location_xmax or \
                    ymin > location_ymax or \
                    xmax < location_xmin or \
                    ymax < location_ymin:
                temp_objects.remove(object)

            if xmin < location_xmin:  xmin = location_xmin;
            if ymin < location_ymin:  ymin = location_ymin;
            if xmax > location_xmax:  xmax = location_xmax;
            if ymax > location_ymax:  ymax = location_ymax;

            object.find('bndbox').find('xmin').text = xmin-location_xmin
            object.find('bndbox').find('ymin').text = ymin-location_ymin
            object.find('bndbox').find('xmax').text = xmax-location_xmin
            object.find('bndbox').find('ymax').text = ymax-location_ymin

        mask = np.zeros([int(info["height"]), int(info["width"]), len(temp_objects)],
                        dtype=np.uint8)

        for i, object in enumerate(temp_objects):
            xmin = int(float(object.find('bndbox').find('xmin').text))
            ymin = int(float(object.find('bndbox').find('ymin').text))
            xmax = int(float(object.find('bndbox').find('xmax').text))
            ymax = int(float(object.find('bndbox').find('ymax').text))

            Y = np.array([ymin, ymax, ymax, ymin, ])
            X = np.array([xmin, xmin, xmax, xmax])

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(Y, X)
            mask[rr, cc, i] = 1

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(object.find('name').text) for object in temp_objects])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "robomaster":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# In[5]:


# ## Create Model

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# In[7]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


def start_train(imagelocation):
    print('------------------------startrain ' + imagelocation + '------------------------')
    # Training dataset.
    dataset_train = RobomasterDataset()
    dataset_train.load_robomaster(imagelocation, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RobomasterDataset()
    dataset_val.load_robomaster(imagelocation, "val")
    dataset_val.prepare()

    # In[6]:

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # ## Training
    #
    # Train in two stages:
    # 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
    #
    # 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

    # In[8]:

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    # In[9]:

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")


start_train('image_center')
start_train('image_zs')
start_train('image_ys')
start_train('image_zx')
start_train('image_yx')


# In[10]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_robomaster.h5")
model.keras_model.save_weights(model_path)

"""



# ## Detection

# In[11]:


class InferenceConfig(RobomasterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# In[12]:


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                   image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

# In[13]:


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# ## Evaluation

# In[14]:

# Validation dataset
dataset_test = RobomasterDataset()
dataset_test.load_robomaster(ROBOMASTER_DIR, "test")
dataset_test.prepare()

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = getImageIds('test.txt')
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, inference_config,
                                                                              image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))

# In[ ]:



"""






