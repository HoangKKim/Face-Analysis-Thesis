import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import yaml
import cv2
import math
import os.path as osp
import numpy as np

import os
import json

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import post_process_batch
from utils.augmentations import letterbox

def load_single_image(frame, path, img_size=640, stride=32, auto=True):
    """
    Load an image, resize with letterbox, convert BGR to RGB, HWC to CHW.
    """
    
    if frame is not None:        
        img0 = frame
    else:
        img0 = cv2.imread(path)  
        assert img0 is not None, f'Image Not Found: {path}'

    # Resize với letterbox
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]

    # Convert BGR -> RGB, HWC -> CHW
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return path, img, img0

def detector(frame, img_path, data, weights, imgsz = 1024, device = 0, conf_thres = 0.7, iou_thres = 0.5, 
             match_iou = 0.6, scales = [1], line_thick = 2, counting = 0, num_offsets = 2):
    detected_objects = []

    with open(data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(device, batch_size=1)
    # print('Using device: {}'.format(device))

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride

    
    # add thêm các giá trị thresholdthreshold
    data['conf_thres_part'] = conf_thres  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = iou_thres  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = match_iou  # whether a body-part in matched with one body bbox

    path, img, im0 = load_single_image(frame, img_path)    
    org_img = im0.copy()
        
        
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    out_ori = model(img, augment=True, scales=scales)[0]
    body_dets = non_max_suppression(out_ori, conf_thres, iou_thres, 
                                    classes=[0], num_offsets=num_offsets)
    part_dets = non_max_suppression(out_ori, conf_thres, iou_thres, 
                                    classes=list(range(1, 1 + num_offsets//2)), 
                                    num_offsets=num_offsets)
        
    # Post-processing of body and part detections
    bboxes, points, scores, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], 
                                                         body_dets, part_dets)

    # line_thick = max(im0.shape[:2]) // 1280 + 3
    line_thick = max(im0.shape[:2]) // 1000 + 3
        
    instance_counting = 0
    

    for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
        [x1, y1, x2, y2] = bbox                

        f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
            
        # number of person detected
        instance_counting += 1               
            
        cv2.rectangle(org_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), line_thick)    
        cv2.imwrite('./output/output_image/output_frame_{}.jpg'.format(img_path), org_img)
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

    if counting:
        cv2.putText(im0, "Num:"+str(instance_counting), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0,0,255), 2, cv2.LINE_AA)

    return detected_objects

        
            
        
  