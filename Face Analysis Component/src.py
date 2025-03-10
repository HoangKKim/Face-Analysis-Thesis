import cv2
import dlib
import numpy as np

# Load pre-trained models
face_detector = dlib.get_frontal_face_detector()  # Face detection
# face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # CNN face detection
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark detection

# Constants for eye aspect ratio (EAR) and yawning detection
EYE_AR_THRESH = 0.2  # Threshold for eye blinking
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames for blinking
YAWN_THRESH = 50  # Threshold for yawning detection

# Function to calculate 3D head pose
def calculate_head_pose(landmarks):
    # Extract key landmarks
    nose_tip = landmarks[30]
    left_eye_corner = landmarks[36]
    right_eye_corner = landmarks[45]
    left_eyebrow = landmarks[21]
    right_eyebrow = landmarks[22]
    
    # Calculate yaw
    yaw = np.arctan2((right_eye_corner[0] - left_eye_corner[0]), (right_eye_corner[1] - left_eye_corner[1])) - np.arctan2((right_eye_corner[0] - nose_tip[0]), (right_eye_corner[1] - nose_tip[1]))
    
    # Calculate pitch
    pitch = np.arctan2((nose_tip[1] - (left_eye_corner[1] + right_eye_corner[1]) / 2), (nose_tip[0] - (left_eye_corner[0] + right_eye_corner[0]) / 2))
    
    # Calculate roll
    roll = np.arctan2((left_eyebrow[1] - right_eyebrow[1]), (left_eyebrow[0] - right_eyebrow[0]))
    
    return yaw, pitch, roll

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect yawning
def detect_yawning(mouth):
    # Calculate the vertical distance between the upper and lower lips
    d_mouth = np.linalg.norm(mouth[2] - mouth[10])  # Upper and lower lip distance
    return d_mouth > YAWN_THRESH

# Function to detect blinking
def detect_blinking(eye_points, frame_count):
    ear = eye_aspect_ratio(eye_points)
    print(ear)
    if ear < EYE_AR_THRESH:
        frame_count += 1
    else:
        if frame_count >= EYE_AR_CONSEC_FRAMES:
            print("Blink detected!")
        frame_count = 0
    return frame_count

# Function to calculate shoulder width
def calculate_shoulder_width(landmarks):
    # Use landmarks for left and right shoulders (approximate positions)
    left_shoulder = landmarks[0]
    right_shoulder = landmarks[16]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    return shoulder_width

# Function to normalize facial landmarks
def normalize_landmarks(landmarks):
    # Normalize landmarks w.r.t. the midpoint between the eyes
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    midpoint = (left_eye + right_eye) / 2
    normalized_landmarks = landmarks - midpoint
    return normalized_landmarks

# Main function for facial analysis
def facial_analysis(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    for face in faces:
        # rect = face.rect  # Extract rectangle from CNN detection
        # Detect facial landmarks
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        normalize_landmark = normalize_landmarks(landmarks)
        
        yaw, pitch, roll = calculate_head_pose(landmarks)

        left_eye = landmarks[36:42]  # Use raw landmarks first
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]
        
        # Detect blinking
        blink_frame_count = 0  # Define globally
        blink_frame_count = detect_blinking(left_eye, blink_frame_count)
        #blink_frame_count = detect_blinking(right_eye, blink_frame_count)

        # Estimate face orientation (yaw, pitch, roll)
        if detect_yawning(mouth):
            print("Yawning detected!")

        # Draw landmarks on the frame for visualization
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        if detect_yawning(mouth):
            cv2.putText(frame, "Yawning", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        if blink_frame_count >= EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "Blinking", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display head pose angles
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Roll: {roll:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

# Main loop for video processing
def main():
    cap = cv2.VideoCapture(0)  # Use webcam or video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform facial analysis
        output_frame = facial_analysis(frame)

        # Display the output frame
        cv2.imshow("Facial Analysis", output_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()