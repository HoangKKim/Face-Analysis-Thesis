import cv2
import numpy as np
import json
import os
import shutil
import logging
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from cfg.keyframes_extractor_cfg import *
from cfg.recognizer_cfg import *
from cfg.expression_cfg import *

from modules.tracking.tracker import Tracker
from modules.detection.src.detector import Detector
from modules.recognizer.recognizer import FaceRecognizer 
from modules.pose_estimation.src.extractor import Keypoint_Extractor
from modules.pose_estimation.src.inference import Inference
from modules.expression.fer_classifier import *

from utils.logger import *
from src.evaluate_result import * 

def move_images(src_folder, dst_folder, logger):
    for subfolder in ['face', 'body']:
        src_subfolder = os.path.join(src_folder, subfolder)
        dst_subfolder = os.path.join(dst_folder, subfolder)

        if not os.path.exists(src_subfolder):
            print(f"Cannot find {src_subfolder}.")
            continue

        os.makedirs(dst_subfolder, exist_ok=True)

        for filename in os.listdir(src_subfolder):
            src_path = os.path.join(src_subfolder, filename)
            dst_path = os.path.join(dst_subfolder, filename)
            shutil.move(src_path, dst_path)

        logger.info(f"Move all image from {src_folder} to {dst_folder}")

def init_dataframes():
    standard_columns = ['student_id', 'ratio_positive', 'ratio_neutral', 'ratio_negative', 'mean', 'std', 'focus_score', 'normalized_focus_score']
    df = pd.DataFrame(columns=standard_columns)
    return df

