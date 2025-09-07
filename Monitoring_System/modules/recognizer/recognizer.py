import os
import cv2
import torch
import random
import numpy as np

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from cfg.recognizer_cfg import *

from modules.expression.preprocessor import FacePreprocessor 

class FaceRecognizer:
    def __init__(self, embedding_model = EMBEDDING_MODEL, detector = DETECTOR, device=DEVICE, face_preprocessor=None,
                 feature_path = 'database/feature/feature_vectors.npy', 
                 label_path = 'database/feature/labels.txt'):
        self.embedding_model = embedding_model
        self.detector = detector
        self.device = device
        self.face_preprocessor = FacePreprocessor() or face_preprocessor      
        self.database_vectors, self.labels = self.load_database(feature_path, label_path)

    def get_embeddings(self, image):
        # Resize and normalize image
        face_pixels = cv2.resize(image, (160, 160))
        face_pixels = face_pixels.astype('float32') / 255.0
        sample = torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model(sample)
        return embedding.squeeze(0).cpu().numpy()

    def extract_database(self, dataset_folder, 
                         save_feature_path='database/feature/feature_vectors.npy', 
                         save_label_path='database/feature/labels.txt',
                         n_sample = 1000):
        all_vectors = []
        labels = []

        for person_folder in os.listdir(dataset_folder):
            print(f'Folder {person_folder} is processing')

            student_id = LABEL_MAP[person_folder]
            print(f'Folder {person_folder} (mapped to {student_id}) is processing')
            
            person_folder_path = os.path.join(dataset_folder, person_folder, 'face_images')
            
            all_images = [os.path.join(person_folder_path, img) for img in os.listdir(person_folder_path)]
            # print(all_images[:5])
            sampled_images =random.sample(all_images, n_sample)

            for image_path in sampled_images:
                image = cv2.imread(image_path)
                embedding_vector = self.get_embeddings(image)
                all_vectors.append(embedding_vector)
                labels.append(student_id)

        np.save(save_feature_path, np.array(all_vectors))
        with open(save_label_path, 'w') as f:
            f.write("\n".join(labels))
        return all_vectors, labels

    def extract_input(self, input_folder, num_samples = 0.3):
        face_images = [os.path.join(input_folder, f)
                       for f in os.listdir(input_folder)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        num_of_samples = int(num_samples * len(face_images))

        if num_of_samples <= 0: 
            sampled_images = face_images
        else:
            sampled_images = random.sample(face_images, num_of_samples)

        vectors = []
        for image_path in sampled_images:
            image = cv2.imread(image_path)
            if image is None:
                continue
            preprocessed = self.face_preprocessor.preprocess_image(image)
            vec = self.get_embeddings(preprocessed)
            vectors.append(vec)

        return vectors

    def load_database(self, feature_path, label_path):
        features = np.load(feature_path)
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return features, labels

    def calc_distance(self, emb1, emb2):
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]

    def recognize(self, input_folder, threshold=0.75, min_consensus=0.5):
        input_vectors = self.extract_input(input_folder)
        db_vectors = np.array(self.database_vectors)
        
        predicted_labels = []
        predicted_scores = []
        
        for vector in input_vectors:
            similarities = cosine_similarity(db_vectors, vector.reshape(1, -1)).reshape(-1)
            max_index = np.argmax(similarities)
            
            # Use adaptive threshold or multiple thresholds
            if similarities[max_index] >= threshold:
                predicted_labels.append(self.labels[max_index])
                predicted_scores.append(similarities[max_index])
        
        if predicted_labels:
            # Weight votes by confidence scores
            weighted_votes = {}
            for label, score in zip(predicted_labels, predicted_scores):
                weighted_votes[label] = weighted_votes.get(label, 0) + score
            
            best_label = max(weighted_votes, key=weighted_votes.get)
            confidence = weighted_votes[best_label] / sum(predicted_scores)
            
            if confidence >= min_consensus:
                return best_label
        
        return "Unknown"
    
def evaluate_predictions(ground_truth_path, predicted_path):
    def read_labels(path):
        label_dict = {}
        with open(path, 'r') as f:
            for line in f:
                if '-' in line:
                    folder, label = line.strip().split(' - ')
                    label_dict[folder.strip()] = label.strip()
        return label_dict

    # Đọc ground truth và predicted
    gt = read_labels(ground_truth_path)
    pred = read_labels(predicted_path)

    # Tính accuracy
    total = 0
    correct = 0
    for folder in gt:
        if folder in pred:
            total += 1
            if gt[folder] == pred[folder]:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy



if __name__ == '__main__':
    FACE_RECOGNIZER = FaceRecognizer()

    # process database
    # database, labels = FACE_RECOGNIZER.extract_database('database/student_images')
    # print(f"Extract successfully {len(database)} data")
    # print(f"Number of labels is: {len(labels)}")



    # FACE_RECOGNIZER = FaceRecognizer()
    database_folder = ['database/feature/feature_vectors.npy', 'database/feature/labels.txt']
    list_folder = [os.path.join('modules/recognizer/test_data', dir) for dir in os.listdir('modules/recognizer/test_data') if os.path.isdir(os.path.join('modules/recognizer/test_data', dir))]

    # print(list_folder)
    score = 0
    for i in range(10):
        labels = []
        for input_folder in list_folder:
            print(f'[INFO] - "Processing {os.path.basename(input_folder)} .. ')
            label = FACE_RECOGNIZER.recognize(input_folder)
            labels.append(f"{os.path.basename(input_folder)} - {label}")
        
        with open('modules/recognizer/test_data/predicted.txt', 'w') as f:
            f.write("\n".join(labels))


        accuracy = evaluate_predictions("modules/recognizer/test_data/labels.txt", "modules/recognizer/test_data/predicted.txt")
        score += accuracy
    print(f"Avg score: {(score / 10)}")


    