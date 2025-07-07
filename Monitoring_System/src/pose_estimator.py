import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib  
import cv2

from modules.pose_estimation.src.model import BehaviorClassifier  
from modules.pose_estimation.src.extractor import Keypoint_Extractor, Feature_Extractor
from modules.pose_estimation.src.inference import Inference

from cfg.pose_cfg import *

predictor = Inference()
keypoint_extractor = Keypoint_Extractor()
predicted_folder = 'output/output_pose' 
root_folder = 'output/output_tracking/'

final_result = []

for person_dir in os.listdir(root_folder):
    body_dir = os.path.join(root_folder, person_dir + '/body')

    os.makedirs(os.path.join(predicted_folder, person_dir), exist_ok=True)
    result = []

    print(f"Processing {person_dir}")
    for file in os.listdir(body_dir):
        img_file_path = os.path.join(body_dir, file)
        image = cv2.imread(img_file_path)

        behavior = predictor.predict(img_file_path)
        print(f"[RESULT] Image: {img_file_path} | Behavior: {behavior}")

        filename, ext = os.path.splitext(file)
        new_filename = f"{filename}_{behavior}.{ext}"
        save_path = os.path.join(predicted_folder, new_filename)
        cv2.imwrite(save_path, image)
        result.append({
            'image': img_file_path,
            "predicted_behaviour": behavior
        })
    final_result.append({
        'person': person_dir,
        'predicted_result': result
    })

import json 
os.makedirs(os.path.join(predicted_folder, person_dir), exist_ok=True)
with open('output/pose_estiame.json', 'w') as f:
    json.dump(final_result, f, indent= 4)
        

    
