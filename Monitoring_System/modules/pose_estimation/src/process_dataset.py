from modules.pose_estimation.src.extractor import Feature_Extractor, Keypoint_Extractor
from cfg.pose_cfg import *
import numpy as np
import cv2
import os
from glob import glob
import json
from tqdm import tqdm

class Process_Dataset:
    def __init__(self):
        pass

    def gather_labels(self, label_folder, output_file):
        label_list = []
        count = 0
        for file in os.listdir(label_folder):
            if file.endswith('.txt'):
                txt_path = os.path.join(label_folder, file)
                
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    if(len(lines)<1):
                        print(txt_path)
                    for line in lines:
                        if line.strip():
                            label_id = line.strip().split()[0]
                            label_list.append({
                                'image': file.replace('.txt', '.jpg'),
                                'label': label_id
                            })
                        # count+=1
        folder_name = os.path.basename(os.path.normpath(label_folder))
        result = [{
            'folder': folder_name,
            'labels': label_list
        }]
        print(f'Number of items in {folder_name}: {len(label_list)}')

        with open(output_file, 'w') as f:
            json.dump(result, f, indent = 4)
        return output_file
    
    def read_label_folder(self, label_folder):
        import json 

        # read json file
        with open(label_folder, 'r') as f:
            label_data = json.load(f)
        
        print('Number of item in dataset: ',len(label_data[0]['labels']))
        return label_data[0]['labels']

    
    def extract_feature(self, image_folder, label_folder):
        # init keypoint extractor
        keypoint_extractor = Keypoint_Extractor()

        label_data = self.read_label_folder(label_folder)

        features = []
        labels = []

        # load qua cac item trong folder
        for item in tqdm(label_data, desc=f"Extracting from {os.path.basename(image_folder)}", unit="img"):
            image_label = int(item['label'])
            img_name = item['image']
            img_path = os.path.join(image_folder, img_name)

            # print(img_path)
            # read image
            
            img = cv2.imread(img_path)
            if img is None:
                print(img_path)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            # init Pose Estimator
            # pose_estimator = Keypoint_Extractor()
            keypoints = keypoint_extractor.inference(img_path)
            uppon_keypoints = keypoint_extractor.gather_upon_body(keypoints)

            # init Feature Extractor
            feature_extractor = Feature_Extractor(uppon_keypoints)
            image_feature = feature_extractor.extract_feature()
            
            if image_feature is None:
                print(f"Error in extract feature: {img_path}")
                continue

            # add to list
            features.append(image_feature)
            labels.append(image_label)

        print(f'Extracted {len(features)} images done')
        return np.array(features), np.array(labels)
 
            
if __name__ == '__main__':
    dataset_processor = Process_Dataset()
    # process from original dataset
    folder = ['train', 'valid', 'test']

    # for i in folder:
    #     origin_folder = os.path.join('./modules/pose_estimation/dataset/original_labels', i)
    #     output_file = os.path.join('./modules/pose_estimation/dataset/labels', i + '.json')
    #     labels = dataset_processor.gather_labels(origin_folder, output_file)

    # extract features
    for i in folder:
        print(f"Process for {folder}")
        image_folder = os.path.join('./modules/pose_estimation/dataset/images', i)
        label_file = os.path.join('./modules/pose_estimation/dataset/labels', i + '.json')
        features, labels = dataset_processor.extract_feature(image_folder, label_file)
        
        print(f"Features {i}: {len(features)} \nLabels {i}: {len(labels)}")
        np.save(os.path.join(BEHAVIOR_FEATURE_DIR, i + '_features.npy'), features)
        np.save(os.path.join(BEHAVIOR_FEATURE_DIR, i + '_labels.npy'), labels)

    #     print(i,'\n',features)   
    
    X = np.load(os.path.join(BEHAVIOR_FEATURE_DIR, 'train' + "_features.npy"))
    y = np.load(os.path.join(BEHAVIOR_FEATURE_DIR, 'train' + "_labels.npy"))

    print(len(X), len(y))
    print(y)

                    






