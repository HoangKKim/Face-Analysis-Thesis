import os
import cv2
import torch
import random
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from cfg.recognizer_cfg import *
from modules.recognizer.recognizer import FaceRecognizer

# get frame in video 
def get_frame_from_video(list_video_input, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    count = 1
    for vid in list_video_input:

        cap = cv2.VideoCapture(vid)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imwrite(os.path.join(dst_dir, f"frame_{count}.jpg"), frame)
            count +=1
    print(f'Get frame from {list_video_input} successfully')


# get face images for each individua
def prepare_images(image_dir, n_sample = 500):
    # init class
    RECOGNIZER = FaceRecognizer()

    # get n_samples from folder
    for dir in os.listdir(image_dir):
        
        # continue/
        print(f'Process {dir} directory')
        individual_dir = os.path.join(image_dir, dir, 'original_images')
        face_dir = os.path.join(image_dir, dir, 'face_images')
        draw_dir = os.path.join(image_dir, dir, 'drawing_images')       
        os.makedirs(face_dir, exist_ok=True)
        os.makedirs(draw_dir, exist_ok=True)


        # process on each dir of student
        all_images = [img for img in os.listdir(individual_dir)]

        sampled_images =random.sample(all_images, n_sample)

        # crop to face images
        for index, image_path in enumerate(sampled_images):
            image = cv2.imread(os.path.join(individual_dir, image_path))
            detected_result = DETECTOR.detect_face(image, 0.3, 0.5)
            if len(detected_result) != 1:
                print(f'Image {image_path} skipped: {"no face" if len(detected_result)==0 else "multiple faces"}')
                continue
            
            x1, y1, x2, y2 = detected_result[0]['bb_face']
            face_score = detected_result[0]['face_score']
            cropped_face_image = image[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(face_dir, f'face_{index+1}.jpg'), cropped_face_image)
            
            img_drawn = image.copy()
            cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_drawn, f'face - {face_score}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            draw_path = os.path.join(draw_dir, f'draw_{index+1}.jpg')
            cv2.imwrite(draw_path, img_drawn)

if __name__ == "__main__":
    data_anno = [
        {
            'out': 'BuiMinh',
            'lst_in': ["Bui_1.mkv", "Bui_2.mkv"]
        }, 
        {
            'out': 'Kim',
            'lst_in': ["Kim_1.mp4", "Kim_2.mp4"]
        },
        {
            'out': 'MinhTrieu',
            'lst_in': ['Minh.mp4']
        },
        {
            'out': 'Ngoc',
            'lst_in': ['Ngoc_1.mp4', 'Ngoc_2.mp4']
        },
        {
            'out': 'Oanh',
            'lst_in': ['Oanh_1.mp4', 'Oanh_2.mp4']
        },
        {
            'out': 'Tam',
            'lst_in': ['Tam_1.MOV', 'Tam_2.MOV']
        },
        {
            'out': 'Thinh',
            'lst_in': ['Thinh.mp4']
        },
        {
            'out': 'Tran',
            'lst_in': ['Tran.MP4']
        }
    ]

    input_root_dir = 'input/recognition'
    output_root_dir = 'database/image'

    # for element in data_anno:
    #     dst_dir = os.path.join(output_root_dir, element['out'])
    #     list_vid_input = [os.path.join(input_root_dir, vid_path) for vid_path in element['lst_in']]
    #     print({'dst': dst_dir, 'in': list_vid_input})
    #     get_frame_from_video(list_vid_input, dst_dir)
    
    prepare_images('database/student_images')

    # rotate image (if neccessary)
    # img = cv2.imread('database/student_images/Oanh/original_images/frame_1564.jpg')
    # rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imwrite('database/student_images/Oanh/original_images/frame_1564.jpg', rotated_image)