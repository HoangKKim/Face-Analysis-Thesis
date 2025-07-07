import cv2
import numpy as np
import json
import os
import shutil

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


def move_images(src_folder, dst_folder):
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

        print(f"Move all image from {src_folder} to {dst_folder}")

def main():
    video_path = 'input/input_video/sub_demo_01_01.mp4'
    output_video_path = 'output/output_video.mp4'
    database_folder = ['database/feature/feature_vectors.npy', 'database/feature/labels.txt']

    TRACKER = Tracker()
    DETECTOR = Detector()
    FACE_RECOGNIZER = FaceRecognizer()
    POSE_PREDICTOR = Inference()
    FER_PREDICTOR = FERClassifier(root_dir = '')
    FER_PREDICTOR.load_model(FER_MODEL)

    
    output_crop_dir = 'output/output_tracking'
    output_json_path = './output/tracking_results.json'

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec cho .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pbar = tqdm(total=total_frames, desc="🔄 Processing video", unit="frame")
    # Step 1: Read video 
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    # compute para for keyframe extractor
    jumping_step = int(fps / NUM_KEYFRAMES_GET_PER_FRAME)

    frame_id = 0
    all_tracking_results = []

    exception_dir = 'output/exceptions'
    exception_json = []
    os.makedirs(exception_dir, exist_ok=True)
    print(f"Tracking & Extract keyframes")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame_for_crop  = frame.copy()   
    #     frame_for_plot  = frame.copy()

    #     tracked_data = TRACKER.track_objects(frame, DETECTOR, frame_id)


    #     # nếu là keyframes thì mới bắt đầu xử lý và lưu trữ ảnh
    #     if (frame_id % jumping_step == 0) and len(tracked_data) > 0:
    #         last_tracking = tracked_data  
    #         all_tracking_results.append(tracked_data)

    #         for obj in tracked_data["objects"]:
    #             bbox = obj['bbox']
    #             x1, y1, x2, y2 = bbox
    #             label = obj['label']  # face or body
    #             person_id = obj.get('person_id', obj['track_id'])

    #             # Tạo thư mục lưu crop: output_tracking/person_XX/{face,body}/
    #             folder_name = f"person_{int(person_id):02d}"
    #             save_dir = os.path.join(output_crop_dir, folder_name, label)
    #             Path(save_dir).mkdir(parents=True, exist_ok=True)

    #             crop_img = frame_for_crop[y1:y2, x1:x2]
    #             if crop_img is not None and crop_img.size > 0: 
    #                 save_path = os.path.join(save_dir, f"frame_{frame_id:05d}.jpg")
    #                 cv2.imwrite(save_path, crop_img)
    #             else:
    #                 print(f"Exception frame: {frame_id}") 
    #                 cv2.imwrite(os.path.join(exception_dir, f"{frame_id}.jpg"), frame)
    #                 exception_json.append({
    #                     'frame_index': frame_id,
    #                     "inf": obj
    #                 })

    #     # print(f"Processed frame {frame_id}")
    #     # progress = (frame_id / total_frames) * 100
    #     # print(f"🟢 Processed frame {frame_id}/{total_frames} ({progress:.2f}%)")

    #     if last_tracking:
    #         for obj in last_tracking["objects"]:
    #             x1, y1, x2, y2 = obj['bbox']
    #             label = obj['label']
    #             person_id = obj.get('person_id', obj['track_id'])

    #             color = (0, 255, 0) if label == 'face' else (255, 0, 0)
    #             text = f"{label}_ID:{person_id}"

    #             cv2.rectangle(frame_for_plot, (x1, y1), (x2, y2), color, 2)
    #             (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    #             cv2.rectangle(frame_for_plot, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, 1)
    #             cv2.putText(frame_for_plot, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    #     out.write(frame_for_plot)
    #     pbar.update(1)
    #     frame_id += 1
    # cap.release()
    # out.release()

    # # Lưu file json kết quả tracking
    # with open(output_json_path, 'w') as f:
    #     json.dump(all_tracking_results, f, indent=4)

    # # Lưu file json kết quả exception
    # with open(os.path.join(exception_dir, 'exception_inf.json'), 'w') as f:
    #     json.dump(exception_json, f, indent=4)

    # print("\nTracking done, results saved.")

    # # ------------------------------------------ RECOGNITION -------------------------------------------------------
    # # recognized để gom các folder về cùng 1 id

    # # define folders
    # existed_person_dir = []
    # unknow_dir = []
    # for person_dir in os.listdir(output_crop_dir):
    #     # define folders
    #     person_path = os.path.join(output_crop_dir, person_dir)
    #     face_folder = os.path.join(person_path, 'face')

    #     if not os.path.exists(face_folder):
    #         shutil.rmtree(person_path)
    #         print(f'{person_path} is deleted')
    #         continue
        
    #     # lấy 1/10 số frame trong folder
    #     score, label = FACE_RECOGNIZER.recognize(database_folder, face_folder,)


    #     if score >= 0.7:
    #         identified_folder = os.path.join(output_crop_dir, label)

    #         if identified_folder not in existed_person_dir:    
    #             existed_person_dir.append(identified_folder)
    #             os.rename(person_path, identified_folder)
    #             print("Recognized successfully and rename the coressponding folder!")
    #             print(f"Folder {person_path} is belonged to {label} - score: {score}")
    #         else:
    #             move_images(person_path, identified_folder)
    #             shutil.rmtree(person_path)
    #             print(f'{person_path} is deleted after moved all imagesimages')

    #     else: 
    #         unknow_folder = os.path.join(output_crop_dir, f"unknow_{(len(unknow_dir) + 1):02d}")
    #         unknow_dir.append(f"{len(unknow_dir) + 1}")
    #         os.rename(person_path, unknow_folder)
    #         print("Recognized failed!")
    #         print(f"Folder {person_path} is belonged to {unknow_folder} - score: {score}")

    #     print('Existed directory: ', existed_person_dir)

    # ----------------------------------------------- behavior -----------------------------------------------------------
    # duyệt qua từng folder -> đoán behavior ; emotion -> lưu vào json 
    predicted_pose_folder = 'output/output_pose'
    unknow_dir = []
    os.makedirs(predicted_pose_folder, exist_ok= True)

    all_people_results = []
    for person_dir in os.listdir(output_crop_dir):
        if person_dir not in unknow_dir:
            person_path = os.path.join(output_crop_dir, person_dir)

            face_folder = os.path.join(person_path, 'face')
            body_folder = os.path.join(person_path, 'body')
            
            # os.makedirs(os.path.join(predicted_pose_folder, person_dir), exist_ok=True)
            result = []
            print(f"Processing {person_dir}")
            for body_file, face_file in zip(os.listdir(body_folder), os.listdir(face_folder)):
                face_img_path = os.path.join(face_folder, face_file)
                body_img_path = os.path.join(body_folder, body_file)

                # face_img = cv2.imread(face_img_path)
                body_img = cv2.imread(body_img_path)

                # emotion
                predicted_emotion, _ = FER_PREDICTOR.inference(face_img_path)
                emotion = FER_LABEL[predicted_emotion]

                # behavior
                behavior = POSE_PREDICTOR.predict(body_img)
                print(f"[RESULT] Image: {body_img_path} | Behavior: {behavior}")
                result.append({
                    'body_image': body_img_path,
                    "predicted_behaviour": behavior,
                    'face_image': face_img_path,
                    'predicted_emotion': emotion
                })
            # save in json file for each person 
            with open(os.path.join(predicted_pose_folder, f'{person_dir}.json'), 'w') as f:
                json.dump(result, f, indent= 4)

            all_people_results.append({
                'person': person_dir,
                'predicted': result
            })
    
    # save in json for all people 
    with open(os.path.join(predicted_pose_folder, "all_people.json"), 'w') as f:
        json.dump(all_people_results, f, indent=4)
        
if __name__ == "__main__":
    main()

    
