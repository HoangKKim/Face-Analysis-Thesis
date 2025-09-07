from mmcv.image import imread
from mmpose.utils import register_all_modules
register_all_modules()

from mmpose.apis import inference_topdown, init_model

from typing import  Union

from cfg.pose_cfg import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Keypoint_Extractor:
    """
    A class for human pose estimation using MMPose.

    Attributes:
        model_cfg (str): Path to the model configuration file.
        ckpt (str): Path to the pretrained model checkpoint.
        device (str): Device to run the model on ('cuda' or 'cpu').
        model (Any): The initialized pose estimation model.

    Methods:
        build_model():
            Initializes and returns the pose estimation model.
        
        inference(img_path):
            Runs inference on a single image and returns the keypoints.
        
        add_neck_keypoints(keypoints):
            Computes the neck keypoint as the midpoint between the left and right shoulders.
        
        gather_upon_body(keypoints):
            Extends the keypoints with the computed neck point and selects relevant joints
            (nose, neck, left/right shoulder, elbow, wrist) for further processing.
    """
    def __init__(self, model_cfg = MMPOSE_MODEL_CFG, ckpt = MMPOSE_CKPT_PATH, device = 'cuda'):
        
        """
        Initializes the Pose_Estimator class.

        Args:
            model_cfg (str): Path to model config file.
            ckpt (str): Path to checkpoint file.
            device (str): Device to use for inference.
        """

        self.model_cfg = model_cfg
        self.ckpt = ckpt
        self.device = device
        self.model = self.build_model()
    
    def build_model(self):
        """
        Builds and loads the MMPose model.

        Returns:
            Any: Loaded pose estimation model.
        """
        # init model
        model = init_model(self.model_cfg, self.ckpt, device = self.device)    
        return model

    def inference(self, img: Union[np.ndarray, str]):
        """
        Performs pose estimation on a single image.

        Args:
            img: Path to the input image  or Numpy Arrat contains image content.

        Returns:
            np.ndarray: Array of keypoints for detected humans.
        """
        batch_results = inference_topdown(self.model, img)
        # return a list of keypoints per human in each image
        # print(batch_results)
        return batch_results[0].pred_instances.keypoints
    
    def add_neck_keypoints(self, keypoints):
        """
        Computes the neck keypoint by averaging the left and right shoulder keypoints.

        Args:
            keypoints (np.ndarray): Keypoints array of shape (17, 2) or similar.

        Returns:
            np.ndarray: Coordinates of the estimated neck keypoint.
        """
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        neck = (left_shoulder + right_shoulder) / 2.0
        return neck
    
    def gather_upon_body(self, keypoints):
        """
        Filters and returns a subset of keypoints including the nose, neck,
        left/right shoulders, elbows, and wrists.

        Args:
            keypoints (np.ndarray): Array of shape (1, 17, 2) or (17, 2).

        Returns:
            List[np.ndarray]: List of selected keypoints including the computed neck.
        """
        keypoints = keypoints[0]  
        neck_keypoint = self.add_neck_keypoints(keypoints)
        extended_keypoints = np.vstack([keypoints, neck_keypoint])
        
        indices = [0, -1, 5, 6, 7, 8, 9, 10]
        filtered_keypoints = [extended_keypoints[i] for i in indices]
        return filtered_keypoints 
    
# class Feature_Extractor:
#     def __init__(self, keypoints, frame):
#         self.keypoints = keypoints
#         self.frame = frame
#         self.joint_pairs = [(0, 1), (2, 3), (3, 4), (5, 6), (6, 7)]
        
#     def normalize_keypoints(self):
#         """
#         Normalize joint locations based on image dimensions.

#         Args:
#             keypoints (np.ndarray): Array of shape (N, 2) with joint positions (x_i, y_i).
#             image_width (int): Width of the image.
#             image_height (int): Height of the image.

#         Returns:
#             np.ndarray: Normalized keypoints of shape (N, 2).
#         """
#         img_height, img_width = self.frame.shape[:2]

#         if  img_width == 0 or  img_height== 0:
#             raise ValueError("Image width and height must be non-zero.")
#         keypoints = np.array(self.keypoints)
#         # normalized
#         normalized = np.zeros_like(self.keypoints, dtype= np.float32)
#         normalized[:, 0] = keypoints[:, 0] / img_width
#         normalized[:, 1] = keypoints[:, 1] / img_height

#         return normalized
    
#     def compute_join_distances(self, normalized_keypoints, joint_pairs):
#         """
#         Compute Euclidean distances between specified pairs of joints.

#         Args:
#             normalized_keypoints (np.ndarray): Normalized keypoints of shape (N, 2).
#             joint_pairs (list of tuple): List of index pairs (i, j) representing joints A and B.

#         Returns:
#             np.ndarray: Array of distances for each joint pair.
#         """
#         distances= []
#         for (i,j) in joint_pairs:
#             xa, ya = normalized_keypoints[i]
#             xb, yb = normalized_keypoints[j]
#             dist = np.sqrt((xb-xa)**2 + (yb-ya)**2)
#             distances.append(dist)
#         return np.array(distances, dtype= np.float32)
    
#     def compute_bone_agles(self, normalized_keypoints):
#         """
#         Compute the 5 bone angles (φ1 to φ5) as described.

#         Args:
#             normalized_keypoints (np.ndarray): Array of shape (N, 2), normalized keypoints.

#         Returns:
#             np.ndarray: Array of 5 angles in degrees.
#         """

#         def angle_between_vectors(vector_1, vector_2):
#             unit_vector_1 = vector_1 / (np.linalg.norm(vector_1) + 1e-8)
#             unit_vector_2 = vector_2 / (np.linalg.norm(vector_2) + 1e-8)
#             dot_product = np.clip(np.dot(unit_vector_1, unit_vector_2), -1.0, 1.0)
#             angle_rad = np.arccos(dot_product)
#             return np.degrees(angle_rad) 
        
#         vector_1_0 = normalized_keypoints[0] - normalized_keypoints[1]      # nose - neck
#         vector_3_2 = normalized_keypoints[3] - normalized_keypoints[2]      # r_elbow - r_shoulder 
#         vector_4_3 = normalized_keypoints[4] - normalized_keypoints[3]      # r_wrist - r_elbow
#         vector_6_5 = normalized_keypoints[6] - normalized_keypoints[5]      # l_elbow - l_shoulder
#         vector_7_6 = normalized_keypoints[7] - normalized_keypoints[6]      # l_wrist - l_elbow


#         phi1 = angle_between_vectors(vector_1_0, np.array([0, 1]))          # vertical
#         phi2 = angle_between_vectors(vector_3_2, np.array([-1, 0]))         # leftward
#         phi3 = angle_between_vectors(vector_4_3, np.array([0, 1]))          # vertical
#         phi4 = angle_between_vectors(vector_6_5, np.array([1, 0]))          # rightward
#         phi5 = angle_between_vectors(vector_7_6, np.array([0, 1]))          # vertical

#         return np.array([phi1, phi2, phi3, phi4, phi5], dtype=np.float32)
    
#     def extract_feature(self):
#         """
#         Combines normalized keypoints, joint distances, and bone angles into a single feature vector.

#         Returns:
#             np.ndarray: A feature vector of shape (26,).
#         """

#         # normalized keypoints
#         normalized_keypoints = self.normalize_keypoints()
#         flat_keypoints = normalized_keypoints.flatten()         # shape (16,)

#         # compute distances
#         distances = self.compute_join_distances(normalized_keypoints, self.joint_pairs)     # shape (5,)

#         # compute angles
#         angles = self.compute_bone_agles(normalized_keypoints)  # shape (5,)

#         return np.concatenate([flat_keypoints, distances, angles])      # shape (26)

import numpy as np
import math

class Feature_Extractor:
    def __init__(self, keypoints):
        """
        keypoints: list gồm 8 điểm [nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist]
        mỗi điểm là (x, y)
        """
        self.keypoints = keypoints
        self.joint_pairs = [(2, 3), (3, 4), (5, 6), (6, 7), (1, 0)]  # r_shoulder→r_elbow, r_elbow→r_wrist, ...

    def normalize_keypoints_relative_to_body(self):
        """
        Chuẩn hóa tọa độ keypoint theo cổ (neck-centered), scale theo khoảng cách 2 vai.
        Trả về mảng (8,2) hoặc None nếu không đủ điều kiện.
        """
        keypoints = np.array(self.keypoints, dtype=np.float32)

        if keypoints.shape != (8, 2):
            print(f"[Warning] Invalid keypoints shape: {keypoints.shape}")
            return None

        if not np.isfinite(keypoints).all():
            print("[Warning] Keypoints contain invalid (NaN/inf) values.")
            return None

        neck = keypoints[1]
        r_shoulder = keypoints[2]
        l_shoulder = keypoints[5]

        rel_keypoints = keypoints - neck
        shoulder_dist = np.linalg.norm(r_shoulder - l_shoulder)
        if shoulder_dist < 1e-5:
            print("[Warning] Shoulder distance too small.")
            return None

        rel_keypoints /= shoulder_dist
        return rel_keypoints

    def compute_direction_vectors(self, kp):
        """
        Trả về 6 unit vector (12 số) từ các cặp khớp.
        """
        vectors = []
        for i, j in self.joint_pairs:
            v = kp[j] - kp[i]
            norm = np.linalg.norm(v)
            if norm < 1e-5:
                vectors.extend([0, 0])
            else:
                vectors.extend((v / norm).tolist())
        return np.array(vectors)

    def compute_joint_angles_relative(self, kp):
        """
        Tính 5 góc tạo bởi 3 điểm liên tiếp: r_shoulder-elbow-wrist, l_shoulder-elbow-wrist, neck-nose
        """
        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-5)
            return math.acos(np.clip(cos_angle, -1.0, 1.0))  # in radians

        angles = [
            angle(kp[2], kp[3], kp[4]),  # R arm
            angle(kp[5], kp[6], kp[7]),  # L arm
            angle(kp[3], kp[2], kp[1]),  # neck-Rshoulder-elbow
            angle(kp[6], kp[5], kp[1]),  # neck-Lshoulder-elbow
            angle(kp[0], kp[1], (kp[2] + kp[5]) / 2)  # nose-neck-midshoulder
        ]
        return np.array(angles)

    def compute_joint_distances(self, kp, joint_pairs):
        """
        Tính khoảng cách giữa các khớp được chỉ định.
        """
        return np.array([np.linalg.norm(kp[i] - kp[j]) for i, j in joint_pairs])

    def extract_feature(self):
        """
        Trả về vector đặc trưng 1D: 16 (pos) + 12 (dir) + 5 (angle) + 5 (dist) = 38
        """
        kp_rel = self.normalize_keypoints_relative_to_body()
        if kp_rel is None:
            return None

        directions = self.compute_direction_vectors(kp_rel)  # (10,)
        angles = self.compute_joint_angles_relative(kp_rel)  # (5,)
        distances = self.compute_joint_distances(kp_rel, self.joint_pairs)  # (5,)
        flat_kp = kp_rel.flatten()  # (16,)

        return np.concatenate([flat_kp, directions, angles, distances])  # (36,)

if __name__ == '__main__':
    # from modules.pose_estimation.mmpose.mmpose.utils import register_all_modules
    img_path = 'output/official_demo_1/recognition/Student_02/body/frame_12040.jpg'
    # register_all_modules()

    # read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # init Pose Estimator
    pose_estimator = Keypoint_Extractor()
    keypoints = pose_estimator.inference(img_path)
    # print(len(keypoints))
    uppon_keypoints = pose_estimator.gather_upon_body(keypoints)
    print(f"Uppon kpts: {uppon_keypoints}")

    # # init Feature Extractor
    # feature_extractor = Feature_Extractor(uppon_keypoints, img)

    # image_feature = feature_extractor.extract_feature()

    # print(image_feature)


    # plotting keypoints
    for (x, y) in uppon_keypoints:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)  # màu xanh

    # plotting line connection
    for (i, j) in UPPON_SKELETON:
        pt1 = tuple(uppon_keypoints[i].astype(int))
        pt2 = tuple(uppon_keypoints[j].astype(int))
        cv2.line(img, pt1, pt2, (255, 0, 0), 1) 

    # show plotting image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Keypoints + Skeleton (COCO Format)")
    plt.show()

