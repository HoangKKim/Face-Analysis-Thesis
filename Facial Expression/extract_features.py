import cv2 
import dlib
import numpy as np
import os
from scipy.spatial.distance import euclidean

DETECTOR = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
LANDMARK_MODEL = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
OUTPUT_FOLDER = "./Data/landmarks"
FEATURE_VECTOR_FILE = "./Data/landmark_vectors.npy"
FILENAME_INDEX_FILE = "./Data/filenames.txt"
TRAIN_DATA_PATH = "data/train"

def get_distance(p1, p2, vector):
    x1, y1 = vector[p1 * 2], vector[p1 * 2 + 1]
    x2, y2 = vector[p2 * 2], vector[p2 * 2 + 1]
    return euclidean((x1, y1), (x2, y2))

def get_angle(p1, p2, p3, vector):
    x1, y1 = vector[p1 * 2], vector[p1 * 2 + 1]
    x2, y2 = vector[p2 * 2], vector[p2 * 2 + 1]
    x3, y3 = vector[p3 * 2], vector[p3 * 2 + 1]

    a = np.array([x1 - x2, y1 - y2])
    b = np.array([x3 - x2, y3 - y2])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle

def get_aspect_ratio(p1, p2, p3, p4, vector):
    vertical = get_distance(p2, p3, vector)
    horizontal = get_distance(p1, p4, vector)
    return vertical / (horizontal + 1e-6)

def get_centroid_distances(vector):
    coords = np.array(vector).reshape(68, 2)
    centroid = np.mean(coords, axis=0)
    return [euclidean(pt, centroid) for pt in coords]

def get_symmetry_features(vector, pairs):
    return [abs(vector[p1 * 2] - vector[p2 * 2]) for (p1, p2) in pairs]

def extract_landmarks(image_input):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    all_feature_vectors = []
    filenames = []

    # Determine if input is a file or folder
    if os.path.isfile(image_input):
        image_paths = [image_input]
    elif os.path.isdir(image_input):
        image_paths = [
            os.path.join(image_input, f)
            for f in os.listdir(image_input)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    else:
        raise ValueError("Invalid image input: must be a file or directory path.")

    for image_path in image_paths:
        filename = os.path.basename(image_path)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_result = DETECTOR(gray, 1)

        if not detected_result:
            print(f"No face detected in {filename}")
            continue

        if len(detected_result) > 1:
            print(f"Multiple faces detected in {filename}, using the first one.")
            detected_result = [detected_result[0]]

        rect = detected_result[0].rect
        landmarks = LANDMARK_MODEL(gray, rect)

        x_min, y_min = rect.left(), rect.top()
        width, height = rect.width(), rect.height()

        norm = []
        for i in range(68):
            x = (landmarks.part(i).x - x_min) / width
            y = (landmarks.part(i).y - y_min) / height
            norm.extend([x, y])
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)

        dists = [
            get_distance(36, 45, norm),  # 2 eyes
            get_distance(27, 30, norm),  # nose
            get_distance(48, 54, norm),  # mouth
            get_distance(39, 42, norm),  # open eyes
            get_distance(62, 66, norm),  # open mouth
        ]
        
        angles = [
            get_angle(36, 39, 42, norm),  # around left eye
            get_angle(45, 42, 39, norm),  # around right eye
            get_angle(48, 51, 54, norm),  # mouth bottom
            get_angle(31, 27, 35, norm),  # nose bridge
        ]
        
        eye_ar_left = get_aspect_ratio(36, 37, 41, 39, norm)
        eye_ar_right = get_aspect_ratio(42, 43, 47, 45, norm)
        mouth_ar = get_aspect_ratio(48, 51, 57, 54, norm)

        symmetry_pairs = [
            (36, 45), (37, 44), (38, 43), (39, 42),  # eyes
            (31, 35), (32, 34),  # nose
            (48, 54), (49, 53), (50, 52)  # mouth
        ]
        symmetry = get_symmetry_features(norm, symmetry_pairs)
        
        centroid_distances = get_centroid_distances(norm)

        ratio = [
            dists[0] / dists[1],  # eyes to nose
            dists[2] / get_distance(27, 33, norm),  # mouth to nose
        ]
        
        coords = np.array(norm).reshape(68, 2)
        delta_features = []
        for i in range(67):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            delta_features.extend([dx, dy])

        feature_vector = (np.array(norm).flatten().tolist() + dists + ratio + angles + [eye_ar_left, eye_ar_right, mouth_ar] + symmetry + centroid_distances + delta_features)
        all_feature_vectors.append(feature_vector)
        filenames.append(filename)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, image)
        print(f"Processed {filename}")

    return all_feature_vectors

if __name__ == '__main__':
    database_folder = [os.path.join(TRAIN_DATA_PATH, folder) for folder in os.listdir(TRAIN_DATA_PATH) if os.path.isdir(os.path.join(TRAIN_DATA_PATH, folder))]
    FEATURE_FILE = './Data/feature_vectors/features.npy'
    LABEL_FILE = './Data/feature_vectors/labels.txt'
    all_features = []
    all_labels = []
    # extract feature in database
    for folder in database_folder:
        features = extract_landmarks(folder)
    
        if not features:
            print(f"No valid features for {folder}")
            continue
        
        label = os.path.basename(folder)
        all_features.extend(features)
        all_labels.extend([os.path.basename(folder)] * len(features))
        print(f"Extract feature vector for {folder}")

    # save database and its id in file (binary or npy)
    np.save(FEATURE_FILE, np.array(all_features))
    with open(LABEL_FILE, "w") as f:
        f.writelines([str(name) + "\n" for name in all_labels])
