import cv2
import numpy as np
import json

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from modules.tracking.tracker import Tracker
from modules.detection.src.detector import Detector
from src.tracker import tracking

if __name__ == "__main__":
    # detecting and tracking from input video
    video_path = 'input/input_video/215475_class.mp4'
    tracker = Tracker(CONF_THRESHOLD, MAX_AGE, N_INIT, CLASS_NAMES, CLASS_ID)
    detector = Detector(video_path, DATA, WEIGHTS, IMG_SIZE, DEVICE, CONF_THRES, IOU_THRES, MATCH_IOU, SCALES, LINE_THICK, COUNTING, NUM_OFFSETS)
    output_video_path = './output/output_video/output_result.mp4'
    output_json_path = './output/tracking_results.json'

    tracking(video_path, tracker, detector, output_json_path, output_video_path)


    