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

from utils.torch_utils import select_device
from utils.general import non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import post_process_batch
from utils.augmentations import letterbox
from cfg.detector_cfg import *


class Detector:
    def __init__(self, img_path = None, data = DATA, weights = WEIGHTS, imgsz = 1024, device = DEVICE, body_conf_thres = BODY_CONF_THRES, face_conf_thres = FACE_CONF_THRES, 
                 iou_thres = IOU_THRES, match_iou = MATCH_IOU, scales = [1], line_thick = LINE_THICK, counting = COUNTING, num_offsets = NUM_OFFSETS):
        self.img_path = img_path
        # self.frame = cv2.imread(img_path)
        self.data = data
        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self.body_conf_thres = body_conf_thres
        self.face_conf_thres = face_conf_thres
        self.iou_thres = iou_thres
        self.match_iou = match_iou
        self.scales = scales
        self.line_thick = line_thick
        self.counting = counting
        self.num_offsets = num_offsets
    
    def load_single_image(self, frame, img_size = 640, stride = 32, auto = True):
        """
        Load an image, resize with letterbox, convert BGR to RGB, HWC to CHW.
        """

        img0 = frame
        
        # Resize với letterbox
        img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

        # Convert BGR -> RGB, HWC -> CHW
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return self.img_path, img, img0

    def detect_objects(self, frame, body_conf_thres = BODY_CONF_THRES, face_conf_thres = FACE_CONF_THRES):
        """get both face and body"""
        detected_objects = []
        self.body_conf_thres = body_conf_thres
        self.face_conf_thres = face_conf_thres
        with open(self.data) as f:
            data = yaml.safe_load(f)  # load data dict

        device = select_device(self.device, batch_size=1)

        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())  # model stride

        # add rhreshold values into data
        data['conf_thres_part'] = self.body_conf_thres     # the larger conf threshold for filtering body-part detection proposals
        data['iou_thres_part'] = self.iou_thres      # the smaller iou threshold for filtering body-part detection proposals
        data['match_iou_thres'] = self.match_iou     # whether a body-part in matched with one body bbox

        # load frame need to be processed
        path, img, im0 = self.load_single_image(frame = frame, stride = stride)    
        org_img = im0.copy()
            
        # process on this frame
        img = torch.from_numpy(img).to(device)
        img = img / 255.0           # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]         # expand for batch dim

        out_ori = model(img, augment=True, scales=self.scales)[0]
        body_dets = non_max_suppression(out_ori, self.body_conf_thres, self.iou_thres, 
                                        classes=[0], num_offsets=self.num_offsets)
        part_dets = non_max_suppression(out_ori, self.face_conf_thres, self.iou_thres, 
                                        classes=list(range(1, 1 + self.num_offsets//2)), 
                                        num_offsets=self.num_offsets)
            
        # Post-processing of body and part detections
        bboxes, points, scores, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], 
                                                            body_dets, part_dets)
            
        instance_counting = 0
        
        for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            [x1, y1, x2, y2] = bbox                

            f_score, f_bbox = point[0][2], point[0][3:]         # bbox format [x1, y1, x2, y2]
                
            # number of person detected
            instance_counting += 1               
            
            if f_score != 0:
                [px1, py1, px2, py2] = f_bbox
                # get info in both face & body 
                object = {
                    'person_id': i,
                    'bb_body': [int(x1), int(y1), int(x2), int(y2)],
                    'bb_face': [int(px1), int(py1), int(px2), int(py2)],
                    'body_score': float('%.2f' % score),
                    'face_score': float('%.2f' % f_score),
                }
                detected_objects.append(object)

            # in case missing face: get only body
            elif f_score == 0:
                object = {
                    'person_id': i,
                    'bb_body': [int(x1), int(y1), int(x2), int(y2)],
                    'bb_face': None,
                    'body_score': float('%.2f' % score),
                    'face_score': float('%.2f' % f_score),
                }
                detected_objects.append(object)

        if self.counting:
            cv2.putText(im0, "Num:"+str(instance_counting), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0,0,255), 2, cv2.LINE_AA)

        return detected_objects
    
    def detect_face(self, frame, body_conf_thres = 0.5, face_conf_thres = 0.7):
        self.body_conf_thres = body_conf_thres
        self.face_conf_thres = face_conf_thres

        
        # get only face information 
        detected_objects = []
        with open(self.data) as f:
            data = yaml.safe_load(f)  # load data dict

        device = select_device(self.device, batch_size=1)

        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())  # model stride

        # add rhreshold values into data
        data['conf_thres_part'] = self.body_conf_thres     # the larger conf threshold for filtering body-part detection proposals
        data['iou_thres_part'] = self.iou_thres      # the smaller iou threshold for filtering body-part detection proposals
        data['match_iou_thres'] = self.match_iou     # whether a body-part in matched with one body bbox

        # load frame need to be processed
        path, img, im0 = self.load_single_image(frame = frame, stride = stride)    
        org_img = im0.copy()
            
        # process on this frame
        img = torch.from_numpy(img).to(device)
        img = img / 255.0           # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]         # expand for batch dim

        out_ori = model(img, augment=True, scales=self.scales)[0]
        body_dets = non_max_suppression(out_ori, self.body_conf_thres, self.iou_thres, 
                                        classes=[0], num_offsets=self.num_offsets)
        part_dets = non_max_suppression(out_ori, self.face_conf_thres, self.iou_thres, 
                                        classes=list(range(1, 1 + self.num_offsets//2)), 
                                        num_offsets=self.num_offsets)
            
        # Post-processing of body and part detections
        bboxes, points, scores, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], 
                                                            body_dets, part_dets)
            
        instance_counting = 0
        
        for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            [x1, y1, x2, y2] = bbox                

            f_score, f_bbox = point[0][2], point[0][3:]         # bbox format [x1, y1, x2, y2]
                
            # number of person detected
            instance_counting += 1               
            
            if f_score != 0:
                [px1, py1, px2, py2] = f_bbox

                object = {
                    'person_id': i,
                    'bb_body': [int(x1), int(y1), int(x2), int(y2)],
                    'bb_face': [int(px1), int(py1), int(px2), int(py2)],
                    'body_score': float('%.2f' % score),
                    'face_score': float('%.2f' % f_score),
                }
                detected_objects.append(object)

        if self.counting:
            cv2.putText(im0, "Num:"+str(instance_counting), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0,0,255), 2, cv2.LINE_AA)

        return detected_objects





        
            
        
  