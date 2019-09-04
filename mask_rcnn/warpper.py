import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

moduleBase = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../'))
if not moduleBase in sys.path:
    sys.path.append(moduleBase)

from utils import *
from mask_rcnn.mrcnn import utils,visualize
from mask_rcnn.mrcnn import model as modellib
from mask_rcnn import coco

MODEL_DIR = os.path.join('./tmp', "logs")
COCO_MODEL_PATH = auto_download('tmp/weights/','maskrcnn_coco')

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

def predict(frame_bgr):
    return model.detect([frame_bgr], verbose=0)[0]

def plot(detections,frame_bgr):
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i,roi in enumerate(rois):
        y1,x1,y2,x2 = roi
        cv2.rectangle(overlap,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(overlap,class_names[class_ids[i]],(x1,y1+25), font, 1,(255,255,255),2,cv2.LINE_AA)
    return overlap

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()
predict(np.zeros((512,512,3),dtype=np.uint8))
print('> Done')