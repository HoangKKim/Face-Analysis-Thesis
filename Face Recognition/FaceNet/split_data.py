from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import shutil

def split_data(data, labels, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42):
    """
    Splits the data into training, validation, and test sets.
    :param input_file: Path to the input .npz file containing the data.
    :param train_file: Path to save the training data.
    :param test_file: Path to save the test data.  
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, train_size=train_size, random_state=random_state, stratify=labels)
    
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_images(images, labels, split_name, label_names, output_base_dir):
    for idx, img in enumerate(images):
        label = labels[idx]
        class_name = label_names[label]
        class_dir = os.path.join(output_base_dir, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)
        file_name = f'{class_name}_{idx}.jpg'
        cv2.imwrite(os.path.join(class_dir, file_name), img)

if __name__ == '__main__':
    labels_map = {
        'Kim': 0,
        'Oanh': 1,
        'Minh': 2,
        'Ngoc': 3,
    }
    
    data = []
    labels = []
    label_names = {v: k for k, v in labels_map.items()}
    
    img_path = './raw_data/data'
    output_base_dir = 'output_data'
    
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
        
    for split in ['train', 'val', 'test']:
        for name in labels_map.keys():
            os.makedirs(os.path.join(output_base_dir, split, name), exist_ok=True)

    
    for person, label in labels_map.items():
        person_folder = os.path.join(img_path, person)
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            
            img = cv2.imread(file_path)
            if img is None:
                continue
            
            img = cv2.resize(img, (160, 160))
            data.append(img)
            labels.append(label)
            
    data = np.array(data)
    labels = np.array(labels)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, labels)
    
    save_images(X_train, y_train, 'train', label_names, output_base_dir)
    save_images(X_val, y_val, 'val', label_names, output_base_dir)
    save_images(X_test, y_test, 'test', label_names, output_base_dir)