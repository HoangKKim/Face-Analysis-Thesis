from modules.detection.src.detector import Detector
from cfg.detector_cfg import *

import argparse
import json
import os
import cv2

img_path = 'input/input_images/people.jpg'

# init detector
detector = Detector(img_path)
# detect objects in image
detected_result = detector.detect_objects(cv2.imread(img_path))
print(detected_result)

# write in json file
# with open(f'output/{os.path.splitext(os.path.basename(args.img_path))[0]}_output.json', 'w') as f:
    # json.dump(detected_objs, f, indent = 4)
# print(f"Detected done - The result is written in output/{os.path.splitext(os.path.basename(args.img_path))[0]}_output.json")

    

