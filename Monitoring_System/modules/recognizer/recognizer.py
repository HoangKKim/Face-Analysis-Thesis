import os
import cv2
import torch
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cfg.recognizer_cfg import *

class FaceRecognizer:
    def __init__(self, embedding_model = EMBEDDING_MODEL, detector = DETECTOR, device=DEVICE):
        self.embedding_model = embedding_model
        self.detector = detector
        self.device = device

    def get_embeddings(self, image):
        # Resize and normalize image
        face_pixels = cv2.resize(image, (160, 160))
        face_pixels = face_pixels.astype('float32') / 255.0
        sample = torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.embedding_model(sample)
        return embedding.squeeze(0).cpu().numpy()

    def extract_database(self, dataset_folder, save_feature_path='database/feature/feature_vectors.npy', save_label_path='database/labels.txt'):
        all_mean_vectors = []
        labels = []

        for person_folder in os.listdir(dataset_folder):
            print(f'Folder {person_folder} is processing')
            vectors = []
            person_folder_path = os.path.join(dataset_folder, person_folder)

            for image in os.listdir(person_folder_path):
                if not image.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                image_path = os.path.join(person_folder_path, image)
                image = cv2.imread(image_path)

                detected_result = self.detector.detect_objects(image)
                if len(detected_result) != 1:
                    print(f'Image {image_path} skipped: {"no face" if len(detected_result)==0 else "multiple faces"}')
                    continue

                x1, y1, x2, y2 = detected_result[0]['bb_face']
                cropped_face_image = image[y1:y2, x1:x2]

                embedding_vector = self.get_embeddings(cropped_face_image)
                vectors.append(embedding_vector)

            if not vectors:
                continue

            mean_vector = np.mean(np.array(vectors), axis=0)
            all_mean_vectors.append(mean_vector)
            labels.append(person_folder)
            print(f'Folder {person_folder} is extracted done')

        np.save(save_feature_path, np.array(all_mean_vectors))
        with open(save_label_path, 'w') as f:
            f.write("\n".join(labels))
        return all_mean_vectors, labels

    def extract_input(self, input_folder, num_of_samples=1):
        face_images = [os.path.join(input_folder, f)
                       for f in os.listdir(input_folder)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        sampled_images = random.sample(face_images, min(num_of_samples, len(face_images)))
        vectors = [self.get_embeddings(cv2.imread(image_path)) for image_path in sampled_images]
        mean_vector = np.mean(np.array(vectors), axis=0)
        return mean_vector

    def load_database(self, feature_path, label_path):
        features = np.load(feature_path)
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return features, labels

    def calc_distance(self, emb1, emb2):
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]

    def recognize(self, database_folder_paths, input_folder, num_samples=50):
        feature_path, label_path = database_folder_paths
        database_vectors, labels = self.load_database(feature_path, label_path)
        input_vector = self.extract_input(input_folder, num_samples)
        distances = [self.calc_distance(db_vec, input_vector) for db_vec in database_vectors]

        max_index = np.argmax(distances)
        max_score = distances[max_index]
        matched_label = labels[max_index]

        return max_score, matched_label


if __name__ == '__main__':

    FACE_RECOGNIZER = FaceRecognizer()

    # process database
    # database, labels = FACE_RECOGNIZER.extract_database('database/image')
    # print(database)
    # print(labels)

    FACE_RECOGNIZER = FaceRecognizer()
    database_folder = ['database/feature/feature_vectors.npy', 'database/labels.txt']
    input_folder = 'output/output_tracking/person_01/face'
    min_distance, label, all_distances = FACE_RECOGNIZER.recognize(database_folder, input_folder)

    print(all_distances, min_distance, label)

