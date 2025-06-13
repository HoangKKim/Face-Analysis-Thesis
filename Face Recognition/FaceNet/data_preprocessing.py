import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
import cv2
import shutil

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config_tf)

DETECTOR = MTCNN()
INPUT_FOLDER = './test_data/'
OUTPUT_FOLDER = './preprocessed_data/test'

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = DETECTOR.detect_faces(pixels)
    
    faces = []
    labels = []
    
    if len(results) == 0:
        print(f'No face detected in {filename}')
        return None
    
    for face in results:
        if face['confidence'] < 0.95:
            continue
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)
        face_crop = pixels[y:y+height, x:x+width]
        face_pil = Image.fromarray(face_crop)
        faces.append(face_pil.resize(required_size))
        
    return faces

def extract_all_faces(input_folder):
    all_faces = []
    all_labels = []
    
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)
        
        faces = []
        for file in os.listdir(subdir_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(subdir_path, file)
            face = extract_face(image_path)
            faces.append(face)
            
        labels = [subdir for _ in range(len(faces))]
        
        all_faces.extend(faces)
        all_labels.extend(labels)
    return np.asarray(all_faces), np.asarray(all_labels)

def save_images(images, labels, label_names, output_base_dir):
    for idx, img in enumerate(images):
        label = labels[idx]
        class_name = label_names[label]
        class_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        file_name = f'{class_name}_{idx}.jpg'
        cv2.imwrite(os.path.join(class_dir, file_name), img)
            

if __name__ == '__main__':
    # preprocessed_data = extract_all_faces(INPUT_FOLDER)
    # faces, labels = preprocessed_data
    # print(f'Extracted {len(faces)} faces from {len(labels)} labels.')
    
    # labels_map = {
    #     'Kim': 0,
    #     'Oanh': 1,
    #     'Minh': 2,
    #     'Ngoc': 3,
    # }
    
    # label_names = {v: k for k, v in labels_map.items()}
    # labels = np.array([labels_map[label] for label in labels])
    
    # output_base_dir = OUTPUT_FOLDER
    
    # if os.path.exists(output_base_dir):
    #     shutil.rmtree(output_base_dir)
        
    # for names in labels_map.keys():
    #     os.makedirs(os.path.join(output_base_dir, names), exist_ok=True)
    
    # save_images(faces, labels, label_names, output_base_dir)
    
    for file in os.listdir(INPUT_FOLDER):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        file_path = os.path.join(INPUT_FOLDER, file)
        faces = extract_face(file_path)
        if faces is None:
            continue
        
        output_dir = os.path.join(OUTPUT_FOLDER, os.path.splitext(file)[0])
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, face in enumerate(faces[:2]):
            output_path = os.path.join(output_dir, f'face_{idx}.jpg')
            face.save(output_path)
            print(f'Saved face {idx} from {file} to {output_path}')