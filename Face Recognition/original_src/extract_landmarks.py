import cv2
import dlib
import numpy as np
import os
from scipy.spatial.distance import euclidean

CNN_FACE_DETECTOR = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
LANDMARK_MODEL = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
OUTPUT_FOLDER = "./Data/landmarks"
FEATURE_VECTOR_FILE = "./Data/landmark_vectors.npy"
FILENAME_INDEX_FILE = "./Data/filenames.txt"

def get_distance(p1, p2, vector):
    x1, y1 = vector[p1 * 2], vector[p1 * 2 + 1]
    x2, y2 = vector[p2 * 2], vector[p2 * 2 + 1]
    return euclidean((x1, y1), (x2, y2))

def extract_landmarks(image_folder):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    feature_vectors = []
    filenames = []
    
    for filename in os.listdir(image_folder):
        # get file with image format
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # read image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # detect face 
        # detections = CNN_FACE_DETECTOR(gray, 1)
        
        # if len(detections) != 1:
        #     print(f"Skipping {filename} (found {len(detections)} faces)")
        #     continue
        
        # rect = detections[0].rect
        rect = dlib.rectangle(left = 245, top = 40, right = 394, bottom = 223)
        landmarks = LANDMARK_MODEL(gray, rect)
        
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
        
        feature_vectors = np.array(norm).flatten().tolist() + dists + ratio
        filenames.append(filename)
        
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, image)
        print(f"Processed {filename}")
        
    np.save(FEATURE_VECTOR_FILE, np.array(feature_vectors))
    with open(FILENAME_INDEX_FILE, "w") as f:
        f.writelines([name + "\n" for name in filenames])
        
    print(f"Feature vectors saved to {FEATURE_VECTOR_FILE}")

if __name__ == "__main__":
    image_folder = "./Data/test"
    extract_landmarks(image_folder)