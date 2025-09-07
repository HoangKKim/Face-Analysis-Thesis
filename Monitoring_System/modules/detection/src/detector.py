import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import yaml
import cv2
import os.path as osp
import numpy as np

import os
import json

from ultralytics import YOLO
from cfg.detector_cfg import *

class YOLO_Detector:
    def __init__(self, person_ckpt=PERSON_DETECTION_MODEL, face_ckpt=FACE_DETECTION_MODEL, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_person = YOLO(person_ckpt).to(self.device)
        self.model_face = YOLO(face_ckpt).to(self.device)

    def detect_person(self, image, conf_thres=0.7, iou_thres=0.4, imgsz = 640):
        results = self.model_person(image, imgsz = imgsz ,device=self.device, iou=iou_thres, verbose=False, half = True)[0]

        output = []
        person_id = 0
        max_width_ratio = 0.25
        width = image.shape[1]

        for box in results.boxes:
            cls_id = int(box.cls.item())
            score = box.conf.item()

            if cls_id == 0 and score >= conf_thres:  # class 0 là 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1) > max_width_ratio * width:
                    continue
                score = float(f"{score:.2f}")
                
                output.append({
                    'person_id': person_id,
                    'bb_body': [x1, y1, x2, y2],
                    'body_score': score
                })
                person_id += 1

        return output
    
    def detect_face(self, image, conf_thres=0.6, iou_thres=0.45):
        results = self.model_face(image, device=self.device, iou=iou_thres, verbose=False)[0]

        faces = []
        for box in results.boxes:
            if box.conf.item() >= conf_thres:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                faces.append([x1, y1, x2, y2])
        return faces