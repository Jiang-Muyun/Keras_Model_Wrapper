import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./mask_rcnn")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from .mrcnn import utils,visualize
from .mrcnn import model as modellib
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from .samples.coco import coco


MODEL_DIR = os.path.join('./tmp', "logs")
COCO_MODEL_PATH = os.path.join('./tmp/weights/mask_rcnn', "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

IMAGE_DIR = os.path.join(ROOT_DIR, "images")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
print('> Done')

def mask_rcnn_predict(frame_bgr):
    return model.detect([frame_bgr], verbose=0)[0]

def mask_rcnn_plot(detections,frame_bgr):
    rois = detections['rois']
    masks = detections['masks']
    class_ids = detections['class_ids']
    scores = detections['class_ids']
    
    mask = np.zeros_like(frame_bgr)
    mask_p = np.zeros(frame_bgr.shape[:-1],dtype=np.bool)
    mask_n = np.zeros(frame_bgr.shape[:-1],dtype=np.bool)
    for i,roi in enumerate(rois):
        if class_ids[i] == class_names.index('airplane'):
            mask_p = np.bitwise_or(masks[:,:,i],mask_p)
        else:
            mask_n = np.bitwise_or(masks[:,:,i],mask_n)
    mask[:,:,1] = mask_p.astype(np.uint8)*255
    mask[:,:,2] = mask_n.astype(np.uint8)*255

    overlap = cv2.addWeighted(frame_bgr, 0.5, mask, 0.5, 20)

    for i,roi in enumerate(rois):
        y1,x1,y2,x2 = roi
        if class_ids[i] == class_names.index('airplane'):
            cv2.rectangle(overlap,(x1,y1),(x2,y2),(255,255,0),2)
        else:
            cv2.rectangle(overlap,(x1,y1),(x2,y2),(255,0,255),2)
    return overlap
    

# cap = cv2.VideoCapture('tmp/Videos/9_Very_Close_Takeoffs_Landings.mp4')
# ret, frame_bgr = cap.read()
# frame_bgr = cv2.resize(frame_bgr,(0,0),fx=0.5,fy=0.5)
# with Tick(''):
#     detections = model.detect([frame_bgr], verbose=0)
#     overlap = mask_rcnn_plot(detections,frame_bgr)
#     plt.imshow(cv2.cvtColor(overlap,cv2.COLOR_BGR2RGB));plt.show()