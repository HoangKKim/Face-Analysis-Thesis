import cv2
import dlib
import numpy as np
from scipy.spatial.distance import euclidean

student_vectors = {
    "hari": np.load("./Data/feature_vectors/hari_vector.npy"),
    "hieuthuhai": np.load("./Data/feature_vectors/hieuthuhai_vector.npy"),
}

CNN_FACE_DETECTOR = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
LANDMARK_MODEL = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_distance(p1, p2, vector):
    x1, y1 = vector[p1 * 2], vector[p1 * 2 + 1]
    x2, y2 = vector[p2 * 2], vector[p2 * 2 + 1]
    return euclidean((x1, y1), (x2, y2))

def extract_feature_vector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = CNN_FACE_DETECTOR(gray, 1)
    
    if len(detections) != 1:
        print("Expected exactly one face, found:", len(detections))
        return None
    
    rect = detections[0].rect
    # rect = dlib.rectangle(left = 187, top = 130, right = 219, bottom = 172)
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
        get_distance(36, 45, norm),  # eyes
        get_distance(27, 30, norm),  # nose
        get_distance(48, 54, norm),  # mouth
        get_distance(39, 42, norm),  # open eyes
        get_distance(62, 66, norm),  # open mouth
    ]
    
    ratio = [
        dists[0] / dists[1],
        dists[2] / get_distance(27, 33, norm),
    ]
    
    return np.array(norm).flatten().tolist() + dists + ratio

def recognize_student(image_path):
    image = cv2.imread(image_path)
    input_vector = extract_feature_vector(image)
    
    if input_vector is None:
        print("No valid face detected in the image.")
        return None
    
    min_distance = float('inf')
    matched_student = None
    
    for name, vector in student_vectors.items():
        distance = euclidean(input_vector, vector)
        print(f"Distance to {name}: {distance}")
        if distance < min_distance:
            min_distance = distance
            matched_student = name
            
    print(f"Matched student: {matched_student} with distance {min_distance}")

    if matched_student is None:
        print("No student matched.")
        return None
    return matched_student

if __name__ == "__main__":
    input_path = "./Data/test/test_image_3.jpg"
    recognize_student(input_path)