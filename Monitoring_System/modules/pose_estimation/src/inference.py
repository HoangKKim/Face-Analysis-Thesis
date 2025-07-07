import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib  
import cv2

from modules.pose_estimation.src.model import BehaviorClassifier  
from modules.pose_estimation.src.extractor import Keypoint_Extractor, Feature_Extractor
from cfg.pose_cfg import *

class Inference():
    def __init__(self, model_path=BEHAVIOR_CKPT_PATH, scaler_path=BEHAVIOR_SCALER_PATH, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = CLASS_NAMES
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.keypoint_extractor = Keypoint_Extractor()
    
    def _load_model(self, ckpt_path):
        model = BehaviorClassifier(input_size= 26, num_classes= NUM_CLASSES)
        check_point = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(check_point['model_state_dict'])
        model.to(self.device)
        model.eval()
        print(f"[INFO] Loaded model from {ckpt_path}")
        return model
    
    def _load_scaler(self, scaler_path): 
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"[ERROR] Scaler file is not found at {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        print(f"[INFO] Loaded scaler from {scaler_path}")

        return scaler

    def extract_feature(self, img_path):
        # read image
        if (type(img_path) == str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_path
        
        # init Pose Estimator
        keypoints = self.keypoint_extractor.inference(img_path)
        uppon_keypoints = self.keypoint_extractor.gather_upon_body(keypoints)

        # init Feature Extractor
        feature_extractor = Feature_Extractor(uppon_keypoints, img)
        image_feature = feature_extractor.extract_feature()
        return image_feature

    def preprocess(self, image_feature):
        features = np.array(image_feature).reshape(1, -1)
        return self.scaler.transform(features)
    
    def predict(self, img_path):
        image_feature = self.extract_feature(img_path)
        if image_feature is None:
            raise ValueError('Feature extraction failed')
        
        image_feature = self.preprocess(image_feature)
        input_tensor = torch.tensor(image_feature, dtype= torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            predict_index = torch.argmax(output, dim = 1).item()
        predict_label = self.class_names[predict_index]
        print(f"[INFO] Predicted behavior: {predict_label}")
        return predict_label
    
if __name__ == '__main__':
    predictor = Inference()
    keypoint_extractor = Keypoint_Extractor()

    try:
        behavior = predictor.predict(keypoint_extractor, 'input/sleeping.jpg')
        print(f"[RESULT] Behavior: {behavior}")
    except Exception as e:
        print(f'[ERROR] {e}')

# problem: 

