from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import numpy as np
import os

def calc_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

def recognize_student(avg_embeddings_path, test_embedding_path):
    distances = []
    
    test_data = np.load(test_embedding_path)
    print("Available keys in test_embedding:", test_data.files)
    
    # Adjust 'embedding' if your key is different
    test_embedding = test_data['arr_0']  
    
    # Normalize test embedding
    normalizer = Normalizer(norm='l2')
    test_embedding = normalizer.transform(test_embedding.reshape(1, -1))

    files = os.listdir(avg_embeddings_path)
    for file in files:
        if not file.endswith('.npz'):
            continue
        
        full_path = os.path.join(avg_embeddings_path, file)
        avg_data = np.load(full_path)
        avg_embedding = avg_data['arr_0']
        avg_embedding = normalizer.transform(avg_embedding.reshape(1, -1))
        
        distance = calc_distance(avg_embedding, test_embedding)
        distances.append(distance)

    if not distances:
        print("No valid embeddings found.")
        return

    min_distance = min(distances)
    matched_index = distances.index(min_distance)
    matched_student = files[matched_index].replace('_embedding.npz', '')
    print(f'Matched student: {matched_student} with distance {min_distance}')

    
if __name__ == '__main__':
    avg_embeddings_path = './average_embeddings/'
    test_embedding_path = './test_embedding.npz'
    
    recognize_student(avg_embeddings_path, test_embedding_path)