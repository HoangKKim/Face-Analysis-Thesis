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

    def extract_database(self, dataset_folder, save_feature_path='database/feature/feature_vectors.npy', save_label_path='database/feature/labels.txt'):
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

                detected_result = self.detector.detect_face(image, 0.3, 0.5)
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

    def extract_input(self, input_folder):
        face_images = [os.path.join(input_folder, f)
                       for f in os.listdir(input_folder)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        num_of_samples = int(0.1 * len(face_images))
        if num_of_samples <= 0: 
            sampled_images = face_images
        else:
            sampled_images = random.sample(face_images, num_of_samples)

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

    def recognize(self, database_folder_paths, input_folder):
        feature_path, label_path = database_folder_paths
        database_vectors, labels = self.load_database(feature_path, label_path)
        input_vector = self.extract_input(input_folder)
        distances = [self.calc_distance(db_vec, input_vector) for db_vec in database_vectors]

        max_index = np.argmax(distances)
        max_score = distances[max_index]
        matched_label = labels[max_index]

        return max_score, matched_label


if __name__ == '__main__':

    FACE_RECOGNIZER = FaceRecognizer()

    # process database
    database, labels = FACE_RECOGNIZER.extract_database('database/image')
    print(database)
    print(labels)

    FACE_RECOGNIZER = FaceRecognizer()
    database_folder = ['database/feature/feature_vectors.npy', 'database/feature/labels.txt']
    list_folder = ['Kim', 'Ngoc']
    for i in list_folder:
        input_folder = f'output/output_tracking/{i}/face'

        score, label = FACE_RECOGNIZER.recognize(database_folder, input_folder)
        print(f'Matched label: {label} | Similarity Score: {score:.4f}')
        print(f'Ground truth is: {i}')



# resnet_face_recognition.py
# import os
# import cv2
# import torch
# import random
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from torchvision import transforms
# from PIL import Image
# from torchvision.models import resnet18, ResNet18_Weights
# import torch.nn as nn
# import torch.nn.functional as F

# from cfg.recognizer_cfg import *


# # ========== Embedding Model ==========
# class ResNetEmbeddingModel(nn.Module):
#     def __init__(self, embedding_dim=128):
#         super().__init__()
#         base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(base_model.children())[:-1])
#         self.embedding_head = nn.Linear(512, embedding_dim)

#     def forward(self, x):
#         x = self.backbone(x)       # [B, 512, 1, 1]
#         x = x.view(x.size(0), -1)  # [B, 512]
#         x = self.embedding_head(x)
#         x = F.normalize(x, p=2, dim=1)
#         return x

# # ========== Preprocessing ==========
# def preprocess_image_opencv(cv2_img):
#     """
#     Dùng OpenCV để đọc ảnh và torchvision.transforms mà không dùng PIL
#     """
#     # BGR (OpenCV) → RGB (torchvision expects RGB)
#     img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

#     # Chuyển từ numpy [H, W, C] → torch [C, H, W] & normalize về [0, 1]
#     img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0  # [3, H, W]

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # torchvision Resize hỗ trợ tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     return transform(img_tensor).unsqueeze(0)  # [1, 3, 224, 224]

# # ========== Face Recognizer ==========
# class FaceRecognizer:
#     def __init__(self, detector=DETECTOR, device='cpu'):

#         self.embedding_model = ResNetEmbeddingModel().to(device)
#         self.detector = detector
#         self.device = device

#     def get_embeddings(self, image):
#         sample = preprocess_image_opencv(image).to(self.device)
#         with torch.no_grad():
#             embedding = self.embedding_model(sample)
#         return embedding.squeeze(0).cpu().numpy()

#     def extract_database(self, dataset_folder, save_feature_path='database/feature_vectors.npy', save_label_path='database/labels.txt'):
#         all_mean_vectors = []
#         labels = []

#         for person_folder in os.listdir(dataset_folder):
#             print(f'Processing: {person_folder}')
#             vectors = []
#             person_folder_path = os.path.join(dataset_folder, person_folder)

#             for image_file in os.listdir(person_folder_path):
#                 if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     continue
#                 image_path = os.path.join(person_folder_path, image_file)
#                 image = cv2.imread(image_path)

#                 detected_result = self.detector.detect_face(image, body_conf_thres= 0.3,  face_conf_thres=0.5)
#                 if len(detected_result) != 1:
#                     print(f'Skipped: {image_path}')
#                     continue

#                 x1, y1, x2, y2 = detected_result[0]['bb_face']
#                 cropped = image[y1:y2, x1:x2]
#                 vector = self.get_embeddings(cropped)
#                 vectors.append(vector)

#             if not vectors:
#                 continue

#             mean_vector = np.mean(np.array(vectors), axis=0)
#             all_mean_vectors.append(mean_vector)
#             labels.append(person_folder)
#             print(f'Done: {person_folder}')

#         np.save(save_feature_path, np.array(all_mean_vectors))
#         with open(save_label_path, 'w') as f:
#             f.write("\n".join(labels))
#         return all_mean_vectors, labels

#     def extract_input(self, input_folder):
#         face_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
#                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#         num_of_samples = max(1, int(0.1 * len(face_images)))
#         sampled_images = random.sample(face_images, num_of_samples)
#         vectors = [self.get_embeddings(cv2.imread(path)) for path in sampled_images]
#         return np.mean(np.array(vectors), axis=0)

#     def load_database(self, feature_path, label_path):
#         features = np.load(feature_path)
#         with open(label_path, 'r') as f:
#             labels = [line.strip() for line in f.readlines()]
#         return features, labels

#     def calc_distance(self, emb1, emb2):
#         emb1 = emb1.reshape(1, -1)
#         emb2 = emb2.reshape(1, -1)
#         return cosine_similarity(emb1, emb2)[0][0]

#     def recognize(self, database_paths, input_folder):
#         feature_path, label_path = database_paths
#         db_vectors, labels = self.load_database(feature_path, label_path)
#         input_vector = self.extract_input(input_folder)
#         distances = [self.calc_distance(db_vec, input_vector) for db_vec in db_vectors]

#         max_index = np.argmax(distances)
#         print(f"All distances: {distances}")
#         return distances[max_index], labels[max_index]

# # ========== Run Example ==========
# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     recognizer = FaceRecognizer(device=device)

#     # Step 1: build database (only run once)
#     recognizer.extract_database('database/image')

#     # Step 2: recognize from new folder
#     db_paths = ['database/feature_vectors.npy', 'database/labels.txt']
#     list_folder = ['Kim', 'Ngoc']
#     for i in list_folder:
#         input_folder = f'output/output_tracking/{i}/face'

#         score, label = recognizer.recognize(db_paths, input_folder)
#         print(f'Matched label: {label} | Similarity Score: {score:.4f}')
#         print(f'Ground truth is: {i}')