import cv2 
import dlib
import numpy as np
import os

from cfg.recognizer_cfg import *
from modules.recognizer.recognizer import FaceRecognizer 

# init Recognizer
# RECOGNIZRER = Recognizer()
# num_images_to_sample = 50

root_dir = './output/output_tracking'
FACE_RECOGNIZER = FaceRecognizer()
database_folder = ['database/feature/feature_vectors.npy', 'database/labels.txt']


# identify images in folder
for person_dir in os.listdir(root_dir):
    # define folders
    person_path = os.path.join(root_dir, person_dir)
    face_folder = os.path.join(person_path, 'face')
    
    score, label = FACE_RECOGNIZER.recognize(database_folder, face_folder)

    identified_folder = os.path.join(root_dir, label)
    os.rename(person_path, identified_folder)
    print("Recognized successfully and rename the coressponding folder!")
    print(f"Folder {person_path} is belonged to {label} - score: {score}")



