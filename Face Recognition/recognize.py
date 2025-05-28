import cv2 
import dlib
import numpy as np
import os
from scipy.spatial.distance import euclidean
from detection.src.detector import Detector
from process_dataset import get_distance

MEAN_FEATURE_FILE = './Data/feature_vectors/mean_features.npy'
LABEL_FILE = './Data/feature_vectors/labels.txt'
LANDMARK_MODEL = dlib.shape_predictor("./original_src/shape_predictor_68_face_landmarks.dat")
DETECTOR = Detector()


def load_database(features_file, labels_file):
    features = np.load(features_file)

    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f]
    return features, labels

def extract_input(image, bbox):
    [x1, y1, x2, y2] = bbox
        
    rect = dlib.rectangle(x1, y1, x2, y2)
    landmarks = LANDMARK_MODEL(image, rect)
        
    x_min, y_min = rect.left(), rect.top()
    width = rect.width()
    height = rect.height()
        
    norm = []
    for i in range(68):
        x = (landmarks.part(i).x - x_min) / width   
        y = (landmarks.part(i).y - y_min) / height
        norm.extend([x, y])
            
        # cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
            
    dists = [
        get_distance(36, 45, norm), # 2 eyes
        get_distance(27, 30, norm), # nose
        get_distance(48, 54, norm), # mouth
        get_distance(39, 42, norm), # open eyes
        get_distance(62, 66, norm), # open mouth
    ]
        
    ratio = [
        dists[0] / dists[1],  # eyes to nose
        dists[2] / get_distance(27, 33, norm), # mouth to nose
    ]
    
    feature_vector = np.array(norm).flatten().tolist() + dists + ratio
    return feature_vector


def recognize(input_image, database):
    # detect face
    recognized = []
    detected_result = DETECTOR.detect_objects(input_image)
    [db_features, labels] = database
    for object in detected_result:
        bbox = object['bb_face']
        feature = extract_input(input_image, bbox)
        distances = [euclidean(feature, db_feature) for db_feature in db_features]
        min_index = np.argmin(distances)
        recognized.append({
            'id': labels[min_index],
            'distance': distances[min_index],
            'bbox': bbox 
        })
    return recognized

if __name__ == '__main__':
    input_path = './test_reg_2.jpg'
    input_image = cv2.imread(input_path)

    [db_features, db_labels] = load_database(MEAN_FEATURE_FILE, LABEL_FILE)
    regconized_result = recognize(input_image, [db_features, db_labels])

    print("Regconized result:\n", regconized_result)
    for face in regconized_result:
        [x1, y1, x2, y2] = face['bbox']
        cv2.rectangle(input_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)                
        cv2.putText(input_image, f"ID: {face['id']} - Distance: {str(face['distance'])}", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0,0,255), 2, cv2.LINE_AA)
    cv2.imwrite("recognized_result_2.jpg", input_image)
    # cv2.imshow('recognized result', input_image)





