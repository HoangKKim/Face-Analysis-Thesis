import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config_tf)

DETECTOR = MTCNN()
INPUT_FOLDER = './raw_data/train/'
OUTPUT_FOLDER = './preprocessed_data/'

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = DETECTOR.detect_faces(pixels)
    
    x1, y1, width, height = results[0]['box']
    
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(required_size)
    faces = np.asarray(image)
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
            

if __name__ == '__main__':
    trainX, trainY = extract_all_faces(INPUT_FOLDER)
    print('Extracted faces:', trainX.shape)
    print('Extracted labels:', trainY.shape)
    
    testX, testY = extract_all_faces('./raw_data/test')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    np.savez_compressed(os.path.join(OUTPUT_FOLDER, 'train_data.npz'), trainX, trainY)
    np.savez_compressed(os.path.join(OUTPUT_FOLDER, 'test_data.npz'), testX, testY)