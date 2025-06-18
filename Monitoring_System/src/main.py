import cv2
import numpy as np
import json
import os
from pathlib import Path

from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from cfg.keyframes_extractor_cfg import *
from cfg.recognizer_cfg import *

from modules.tracking.tracker import Tracker
from modules.detection.src.detector import Detector
from modules.recognizer.recognizer import FaceRecognizer 

# def tracking(video_path, tracker, detector, output_json_path, output_video_path, output_crop_dir = 'output/output_tracking'):
def main():
    video_path = 'input/input_video/Kim_Oanh_test_video.mp4'
    output_video_path = 'output/output_video.mp4'
    database_folder = ['database/feature/feature_vectors.npy', 'database/labels.txt']

    TRACKER = Tracker()
    DETECTOR = Detector()
    FACE_RECOGNIZER = FaceRecognizer()

    
    output_crop_dir = 'output/output_tracking'
    output_json_path = './output/tracking_results.json'

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec cho .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Step 1: Read video 
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    # compute para for keyframe extractor
    jumping_step = int(fps / NUM_KEYFRAMES_GET_PER_FRAME)

    frame_id = 0
    all_tracking_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_data = TRACKER.track_objects(frame, DETECTOR, frame_id)
        
        # nếu là keyframes thì mới bắt đầu xử lý và lưu trữ ảnh
        if (frame_id % jumping_step == 0):
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

                # color = (0, 255, 0) if label == 'face' else (255, 0, 0)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # text = f"{label}_ID:{person_id}"
                # (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, 1)
                # cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            print(f"Processed frame {frame_id}")
        frame_id += 1
        # out.write(frame)
    cap.release()
    # out.release()

    # Lưu file json kết quả tracking
    with open(output_json_path, 'w') as f:
        json.dump(all_tracking_results, f, indent=4)

    print("Tracking done, results saved.")

    # recognized để gom các folder về cùng 1 id

    # define folders
    for person_dir in os.listdir(output_crop_dir):
        # define folders
        person_path = os.path.join(output_crop_dir, person_dir)
        face_folder = os.path.join(person_path, 'face')
        
        score, label = FACE_RECOGNIZER.recognize(database_folder, face_folder)

        identified_folder = os.path.join(output_crop_dir, label)
        os.rename(person_path, identified_folder)
        print("Recognized successfully and rename the coressponding folder!")
        print(f"Folder {person_path} is belonged to {label} - score: {score}")

    # step 4: behavior
    #



if __name__ == "__main__":
    main()

    