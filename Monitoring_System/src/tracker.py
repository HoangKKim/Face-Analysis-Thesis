import cv2
import numpy as np
import json
import os
from pathlib import Path

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from modules.tracking.tracker import Tracker
from modules.detection.src.detector import Detector

def tracking(video_path, tracker, detector, output_json_path, output_video_path, output_crop_dir = 'output/output_tracking'):

    # Step 1: Read video 
    cap = cv2.VideoCapture(video_path)
    if(not cap.isOpened()):
        print("Error: Could not open video.")
        exit()

    # get video's information
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # video writter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    all_tracking_results = []

    while True:
        rec, frame = cap.read()
        if not rec:
            break
        origin_frame = frame.copy()

        # Step 2: Track objects on this frame
        tracked_frame = tracker.track_objects(frame, detector, frame_id)

        # Step 3: Save the result in final list
        all_tracking_results.append(tracked_frame)

    #     # Step 4: Visualize on the frame (optional)
    #     for obj in tracked_frame:
    #         x1, y1, x2, y2 = obj['bbox']
    #         track_id = obj['track_id']
    #         label = f"{obj['label']}"
            
    #         label_dir = f'id_{int(track_id):02d}'

    #         color = (0, 255, 0) if obj['label'] == 'face' else (255, 0, 0)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(frame, f"{label} - {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #         # SAVE and CROP image
    #         crop_img = origin_frame[y1:y2, x1:x2]
    #         save_dir = os.path.join(output_crop_dir, label, label_dir)
    #         Path(save_dir).mkdir(parents=True, exist_ok=True)

    #         crop_path = os.path.join(save_dir, f"frame_{frame_id:04d}.jpg")
    #         cv2.imwrite(crop_path, crop_img)

    #     out.write(frame)
        print(f"Processed frame {frame_id}")
        frame_id += 1

    # cap.release()
    # out.release()

    # Step 5: Write tracking result in json file
    with open(output_json_path, 'w') as f:
        json.dump(all_tracking_results, f, indent=4)

    
if __name__ == '__main__':
    video_path = 'input/input_video/215475_class.mp4'
    tracker = Tracker(CONF_THRESHOLD, MAX_AGE, N_INIT, CLASS_NAMES, CLASS_ID)
    detector = Detector(video_path, DATA, WEIGHTS, IMG_SIZE, DEVICE, CONF_THRES, IOU_THRES, MATCH_IOU, SCALES, LINE_THICK, COUNTING, NUM_OFFSETS)
    output_video_path = './output/output_video/output_result.mp4'
    output_json_path = './output/tracking_results.json'

    tracking(video_path, tracker, detector, output_json_path, output_video_path)







