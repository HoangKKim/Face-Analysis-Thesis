import cv2
import numpy as np
import json
import os
from pathlib import Path

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from modules.tracking.tracker import Tracker
from modules.detection.src.detector import Detector

def tracking(video_path, tracker, detector, output_json_path, output_crop_dir='output/output_tracking'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    frame_id = 0
    all_tracking_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_data = tracker.track_objects(frame, detector, frame_id)
        all_tracking_results.append(tracked_data)

        for obj in tracked_data["objects"]:
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            label = obj['label']  # face or body
            person_id = obj.get('person_id', obj['track_id'])

            # Tạo thư mục lưu crop: output_tracking/person_XX/{face,body}/
            folder_name = f"person_{int(person_id):02d}"
            save_dir = os.path.join(output_crop_dir, folder_name, label)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            crop_img = frame[y1:y2, x1:x2]
            save_path = os.path.join(save_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(save_path, crop_img)

        frame_id += 1
        print(f"Processed frame {frame_id}")

    cap.release()

    # Lưu file json kết quả tracking
    with open(output_json_path, 'w') as f:
        json.dump(all_tracking_results, f, indent=4)

    print("Tracking done, results saved.")

    
if __name__ == '__main__':
    video_path = 'input/input_video/Kim_Oanh_test_video.mp4'
    tracker = Tracker(CONF_THRESHOLD, MAX_AGE, N_INIT, CLASS_NAMES, CLASS_ID)
    detector = Detector(video_path, DATA, WEIGHTS, IMG_SIZE, DEVICE, CONF_THRES, IOU_THRES, MATCH_IOU, SCALES, LINE_THICK, COUNTING, NUM_OFFSETS)
    output_video_path = './output/output_video/output_result.mp4'
    output_json_path = './output/tracking_results.json'

    tracking(video_path, tracker, detector, output_json_path)







