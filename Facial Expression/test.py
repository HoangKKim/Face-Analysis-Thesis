import joblib
import numpy as np
from extract_features import extract_landmarks

if __name__ == '__main__':
    svm_model = joblib.load('./data/svm_model.joblib')
    image_path = './data/test/test_disgust.jpg'
    
    features = extract_landmarks(image_path)
    features = np.array(features).reshape(1, -1) if features is not None else None
    
    if features is None:
        print("No valid features extracted from the image.")
        exit()

    if features is not None:
        # Use the first feature vector
        prediction = svm_model.predict(features)
        print(f"Predicted class label: {prediction}")
    else:
        print("No valid features extracted from the image.")