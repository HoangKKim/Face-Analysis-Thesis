import cv2 
import dlib
import numpy as np
import os
from scipy.spatial.distance import euclidean
from detection.src.detector import Detector


DETECTOR = Detector()
LANDMARK_MODEL = dlib.shape_predictor("./original_src/shape_predictor_68_face_landmarks.dat")
OUTPUT_FOLDER = "./Data/landmarks"
FEATURE_VECTOR_FILE = "./Data/landmark_vectors.npy"
FILENAME_INDEX_FILE = "./Data/filenames.txt"

def get_distance(p1, p2, vector):
    x1, y1 = vector[p1 * 2], vector[p1 * 2 + 1]
    x2, y2 = vector[p2 * 2], vector[p2 * 2 + 1]
    return euclidean((x1, y1), (x2, y2))

def extract_landmarks(image_folder):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_feature_vectors = []
    filenames = []
    
    for filename in os.listdir(image_folder):
        # get file with image format
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # read image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detected_result = DETECTOR.detect_objects(frame= image)

        if(len(detected_result) != 1):
            print('invalid dataset')
            continue
        [x1, y1, x2, y2] = detected_result[0]['bb_face']
        
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
            
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
            
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
        all_feature_vectors.append(feature_vector)
        filenames.append(filename)
        
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, image)
        print(f"Processed {filename}")

    return all_feature_vectors

if __name__ == '__main__':
    database_folder = ['database_1', 'database_2', 'database_3']
    MEAN_FEATURE_FILE = './Data/feature_vectors/mean_features.npy'
    LABEL_FILE = './Data/feature_vectors/labels.txt'
    mean_vectors = []
    labels = []
    # extract feature in database
    for id, folder in enumerate(database_folder):
        features = extract_landmarks(folder)
    
        if not features:
            print(f"No valid features for {folder}")
            continue
        # compute mean of feature databases and save in file
        mean_vector = np.mean(np.array(features), axis = 0)
        mean_vectors.append(mean_vector)
        labels.append(id)   

        print(f"Extract mean vector for {folder}")

    # save database and its id in file (binary or npy)
    np.save(MEAN_FEATURE_FILE, np.array(mean_vectors))
    with open(LABEL_FILE, "w") as f:
        f.writelines([str(name) + "\n" for name in labels])
