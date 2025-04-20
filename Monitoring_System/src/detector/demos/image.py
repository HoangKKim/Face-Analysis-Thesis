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

def detector(img_path, data, imgsz, weights, device, conf_thres, iou_thres, match_iou, scales, line_thick, counting, num_offsets = 2):
    detected_objects = []
    colors_list = [
            # [255, 0, 0], [255, 127, 0], [255, 255, 0], [127, 255, 0], [0, 255, 0], [0, 255, 127], 
            # [0, 255, 255], [0, 127, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127],
            [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
            [255, 255, 255],
            [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
            [127, 127, 127],
            [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
            [0, 0, 0],
            [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
        ]  # 27 colors

    with open(data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride

    
    # add thêm các giá trị thresholdthreshold
    data['conf_thres_part'] = conf_thres  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = iou_thres  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = match_iou  # whether a body-part in matched with one body bbox

    images = LoadImages(img_path, img_size=imgsz)
    images_iter = iter(images)

    for index in range(len(images)):
        (single_path, img, im0, _) = next(images_iter)
        org_img = im0.copy()
        if '_res' in single_path or '_vis' in single_path:
            continue
        print(index, single_path, "\n")
        
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
        
        if not os.path.exists('crops'):
            os.makedirs('crops')

        for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            [x1, y1, x2, y2] = bbox            
            
            color = colors_list[i%len(colors_list)]
            # print bb coordinate of body
            print(f'Bounding box-body ({score}): {x1}, {y1}, {x2}, {y2}')
            crop_img = org_img[int(y1):int(y2), int(x1):int(x2)]

            save_path = os.path.join('crops', f'{i}_{os.path.basename(single_path)}')
            cv2.imwrite(save_path, crop_img)

            
            f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
            
            # number of person detected
            instance_counting += 1               
            
            # draw bb for body
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=line_thick)
            if f_score != 0:
                [px1, py1, px2, py2] = f_bbox
                cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=line_thick)
                print(f' + Bounding box - face ({f_score}): {px1}, {py1}, {px2}, {py2}')
                    
                # crop faces
                # crop_img = im0[int(py1):int(py2), int(px1):int(px2)]

                # if not os.path.exists('crops'):
                #     os.makedirs('crops')

                # save_path = os.path.join('crops', f'{i}_{os.path.basename(single_path)}')
                # cv2.imwrite(save_path, crop_img)
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
        
        # save image in the result folder
        if not os.path.exists('result'):
            os.makedirs('result')
        cv2.imwrite(os.path.join("result", os.path.splitext(os.path.basename(single_path))[0]+'.jpg'), im0)

    return detected_objects

        
            
        
  