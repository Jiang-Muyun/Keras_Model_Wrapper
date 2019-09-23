import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import sys
sys.path.append('.')
import my_log as log

config_file = "maskrcnn/configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"

size = (1280, 720)
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
coco_demo = COCODemo(cfg, show_mask_heatmaps=False, min_image_size=800, confidence_threshold=0.6)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture('tmp/S2_Cars_day_cut.mp4')
out = cv2.VideoWriter('tmp/test_S2_Cars_day.avi',fourcc, 20.0, size)

index = 0
while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    frame_bgr = cv2.resize(frame_bgr,size)
    index += 1

    if not ret:
        break
    
    with log.Tick():
        predictions = coco_demo.compute_prediction(frame_bgr)
        top_predictions = coco_demo.select_top_predictions(predictions)

        result = frame_bgr.copy()
        result = coco_demo.overlay_mask(result, top_predictions)
        result = coco_demo.overlay_boxes(result, top_predictions)
        result = coco_demo.overlay_class_names(result, top_predictions)

    cv2.imshow('result',result)
    out.write(result)

    if 32 == cv2.waitKey(1):
        break
out.release()