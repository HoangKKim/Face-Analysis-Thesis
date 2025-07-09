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
from src.supported_functions import *

def main():
    VIDEO_PATH = 'input/input_video/sub_demo_01_01.mp4'
    VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_DIR = os.path.join('output', VIDEO_NAME)
    
    logger = setup_logger(log_path=f'{OUTPUT_DIR}/process_log.txt')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Create {OUTPUT_DIR} successfully")

    OUTPUT_VIDEO_PATH = f'{OUTPUT_DIR}/output_video.mp4'
    DATABASE_DIR = ['database/feature/feature_vectors.npy', 'database/feature/labels.txt']

    # -------------- define modulers ---------------
    TRACKER = Tracker()
    DETECTOR = Detector()
    FACE_RECOGNIZER = FaceRecognizer()
    POSE_PREDICTOR = Inference()
    FER_PREDICTOR = FERClassifier(root_dir = '')
    FER_PREDICTOR.load_model(FER_MODEL)
    EVALUATOR = FocusEvaluator(OUTPUT_DIR, logger)

    # --------------- Define Output's dirs
    OUTPUT_TRACKING_DIR = os.path.join(OUTPUT_DIR,'detection_and_tracking')
    OUTPUT_EXCEPTION_DIR = os.path.join(OUTPUT_DIR,'exceptions')
    OUTPUT_RECOGNIZING_DIR = os.path.join(OUTPUT_DIR, 'recognition')
    OUTPUT_POSE_DIR = os.path.join(OUTPUT_DIR, 'output_pose')
    OUTPUT_FIGURE_DIR = EVALUATOR.get_output_dir()
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, 'total_result.csv')

    # create output folder
    os.makedirs(OUTPUT_TRACKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_EXCEPTION_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RECOGNIZING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_POSE_DIR, exist_ok=True)

    # output_json_path = os.path.join(OUTPUT_DIR,'tracking_results.json')

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec cho .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    pbar = tqdm(total=total_frames, desc="🔄 Processing video", unit="frame")

    # Step 1: Read video 
    if not cap.isOpened():
        logger.error("Cannot open video:", VIDEO_PATH)
        return

    # compute para for keyframe extractor
    jumping_step = int(fps / NUM_KEYFRAMES_GET_PER_FRAME)

    frame_id = 0
    # all_tracking_results = []

    exception_json = []
    os.makedirs(OUTPUT_EXCEPTION_DIR, exist_ok=True)

    logger.info(f"Starting to process video")
    
    # --------------------------------------------------------------- detect & tracking ----------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_for_crop  = frame.copy()   
        frame_for_plot  = frame.copy()

        tracked_data = TRACKER.track_objects(frame, frame_id)

        # nếu là keyframes thì mới bắt đầu xử lý và lưu trữ ảnh
        if (frame_id % jumping_step == 0) and len(tracked_data) > 0:
            last_tracking = tracked_data  
            # all_tracking_results.append(tracked_data)

            for obj in tracked_data["objects"]:
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                label = obj['label']  # face or body
                person_id = obj.get('person_id', obj['track_id'])

                # Tạo thư mục lưu crop: output_tracking/person_XX/{face,body}/
                folder_name = f"person_{int(person_id):02d}"
                save_dir = os.path.join(OUTPUT_TRACKING_DIR, folder_name, label)
                Path(save_dir).mkdir(parents=True, exist_ok=True)

                crop_img = frame_for_crop[y1:y2, x1:x2]
                if crop_img is not None and crop_img.size > 0: 
                    save_path = os.path.join(save_dir, f"frame_{frame_id:05d}.jpg")
                    cv2.imwrite(save_path, crop_img)
                else:
                    cv2.imwrite(os.path.join(OUTPUT_EXCEPTION_DIR, f"{frame_id}.jpg"), frame)
                    exception_json.append({
                        'frame_index': frame_id,
                        "inf": obj
                    })

        if last_tracking:
            for obj in last_tracking["objects"]:
                x1, y1, x2, y2 = obj['bbox']
                label = obj['label']
                person_id = obj.get('person_id', obj['track_id'])

                color = (0, 255, 0) if label == 'face' else (255, 0, 0)
                text = f"{label}_ID:{person_id}"

                cv2.rectangle(frame_for_plot, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame_for_plot, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, 1)
                cv2.putText(frame_for_plot, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame_for_plot)
        pbar.update(1)
        frame_id += 1
    cap.release()
    out.release()

    # # Lưu file json kết quả tracking
    # with open(output_json_path, 'w') as f:
    #     json.dump(all_tracking_results, f, indent=4)

    # Lưu file json kết quả exception
    with open(os.path.join(OUTPUT_EXCEPTION_DIR, 'exception_inf.json'), 'w') as f:
        json.dump(exception_json, f, indent=4)

    logger.info("Tracking done, results saved.")

    # ------------------------------------------ recognized face id & refractor output -------------------------------------------------------
    logger.info("Move to Recognition")

    shutil.copytree(OUTPUT_TRACKING_DIR, OUTPUT_RECOGNIZING_DIR, dirs_exist_ok=True)

    # define folders
    existed_person_dir = []
    unknown_dir = []
    
    for folder in os.listdir(OUTPUT_RECOGNIZING_DIR):
        # define folders
        person_dir = os.path.join(OUTPUT_RECOGNIZING_DIR, folder)
        face_dir = os.path.join(person_dir, 'face')

        if not os.path.exists(face_dir):
            shutil.rmtree(person_dir)
            logger.info(f'{person_dir} is deleted')
            continue
        
        # lấy 1/10 số frame trong folder
        score, label = FACE_RECOGNIZER.recognize(DATABASE_DIR, face_dir,)

        if score >= 0.6:
            identified_folder = os.path.join(OUTPUT_RECOGNIZING_DIR, label)

            if identified_folder not in existed_person_dir:    
                existed_person_dir.append(identified_folder)
                os.rename(person_dir, identified_folder)
                logger.info("Recognized successfully and rename the coressponding folder!")
                logger.info(f"Folder {person_dir} is belonged to {label} - score: {score}")
            else:
                move_images(person_dir, identified_folder, logger)
                shutil.rmtree(person_dir)
                logger.info(f'{person_dir} is deleted after moved all images')

        else: 
            unknow_folder = os.path.join(OUTPUT_RECOGNIZING_DIR, f"unknown_{(len(unknown_dir) + 1):02d}")
            unknown_dir.append(f"unknow_{(len(unknown_dir) + 1):02d}")
            os.rename(person_dir, unknow_folder)
            logger.error("Recognized failed!")
            logger.info(f"Folder {person_dir} is belonged to {unknow_folder} - score: {score}")

        logger.info(f'Existed directory: {existed_person_dir}')

    # ----------------------------------------------- behavior -----------------------------------------------------------
    # duyệt qua từng folder -> đoán behavior ; emotion -> lưu vào json 
    logger.info("Predicting emotions and behaviours")

    # all_people_results = []

    for person_dir in os.listdir(OUTPUT_RECOGNIZING_DIR):
        if person_dir not in unknown_dir:
            person_path = os.path.join(OUTPUT_RECOGNIZING_DIR, person_dir)

            face_folder = os.path.join(person_path, 'face')
            body_folder = os.path.join(person_path, 'body')
            
            result = []
            logger.info(f"Processing {person_dir}")
            for body_file in os.listdir(body_folder):
                face_img_path = os.path.join(face_folder, body_file)
                body_img_path = os.path.join(body_folder, body_file)

                # face_img = cv2.imread(face_img_path)
                body_img = cv2.imread(body_img_path)

                # emotion
                # check if path is existed
                if os.path.exists(face_img_path):
                    predicted_emotion, _ = FER_PREDICTOR.inference(face_img_path)
                    emotion = FER_LABEL[predicted_emotion]
                else:
                    emotion = 'Unknown'

                # behavior
                behavior = POSE_PREDICTOR.predict(body_img)
                result.append({
                    'body_image': body_img_path,
                    "predicted_behaviour": behavior,
                    'face_image': face_img_path,
                    'predicted_emotion': emotion
                })
            # save in json file for each person 
            with open(os.path.join(OUTPUT_POSE_DIR, f'{person_dir}.json'), 'w') as f:
                json.dump(result, f, indent= 4)

            # all_people_results.append({
            #     'person': person_dir,
            #     'predicted': result
            # })
    
    # save in json for all people 
    # with open(os.path.join(OUTPUT_POSE_DIR, "all_people.json"), 'w') as f:
    #     json.dump(all_people_results, f, indent=4)

    # ----------------------------------------------- draw & evaluate -----------------------------------------------------------
    logger.info("Analyzing emotion & behavior")

    # create Dataframes to contain all result 
    FINAL_DATAFRAME = init_dataframes()


    for file in os.listdir(OUTPUT_POSE_DIR):
        file_path = os.path.join(OUTPUT_POSE_DIR, file)
        output_path = os.path.join(OUTPUT_FIGURE_DIR, os.path.splitext(os.path.basename(file))[0])

        # get inf
        EVALUATOR.update_output_dir(output_path)
        EVALUATOR.load_report(file_path)
        fusion, behavior, emotion = EVALUATOR.classify()

        # draw each figure
        EVALUATOR.draw_single_curve(emotion, 'emotion')
        EVALUATOR.draw_single_curve(behavior, 'behavior')
        EVALUATOR.draw_single_curve(fusion, 'fusion')

        # draw all in one
        EVALUATOR.draw_three_curves()

        # compute focus score
        focus_score_result = EVALUATOR.compute_focus_score()
        focus_score_result['student_id'] = os.path.splitext(os.path.basename(file))[0]
        FINAL_DATAFRAME = pd.concat([FINAL_DATAFRAME, pd.DataFrame([focus_score_result])], ignore_index=True)

    # sort by person_id
    FINAL_DATAFRAME = FINAL_DATAFRAME.sort_values(by="student_id", ascending=False)
    # save in csv file
    FINAL_DATAFRAME.to_csv(OUTPUT_CSV_PATH, index=False)
    
        
if __name__ == "__main__":
    main()

    
