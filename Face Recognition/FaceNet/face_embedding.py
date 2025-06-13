import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import os
import cv2

DATA_DIR = './preprocessed_data/'
OUTPUT_DIR = './average_embeddings/'

labels_map = {
    'Kim': 0,
    'Oanh': 1,
    'Minh': 2,
    'Ngoc': 3,
}

# Define embedding extractor
def get_embeddings(model, face_pixels):
    face_pixels = face_pixels.astype('float32') / 255.0
    sample = torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        embedding = model(sample)
    return embedding.squeeze(0).cpu().numpy()

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print("Model loaded.")

    # for person in labels_map.keys():
    #     class_dir = os.path.join(DATA_DIR, person)
    #     embeddings = []

    #     for file_name in os.listdir(class_dir):
    #         if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
    #             continue
    #         file_path = os.path.join(class_dir, file_name)
    #         img = cv2.imread(file_path)
    #         if img is None:
    #             continue
    #         img = cv2.resize(img, (160, 160))
    #         embedding = get_embeddings(model, img)
    #         embeddings.append(embedding)

    #     if len(embeddings) == 0:
    #         print(f"No embeddings found for {person}")
    #         continue

    #     embeddings = np.array(embeddings)
    #     mean_embedding = np.mean(embeddings, axis=0)
    #     output_path = os.path.join(OUTPUT_DIR, f'{person}_embedding.npz')
    #     np.savez_compressed(output_path, mean_embedding)
    #     print(f'Saved mean embedding for {person} to {output_path}')
    
    test_image_path = os.path.join(DATA_DIR, './test/frame_1/face_1.jpg')
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Could not read image: {test_image_path}")
    else:
        img = cv2.resize(img, (160, 160))
        embedding = get_embeddings(model, img)
        output_path = './test_embedding.npz'
        np.savez_compressed(output_path, embedding)
        print(f'Saved test embedding to {output_path}')