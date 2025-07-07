import os
import copy
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from cfg.expression_cfg import *

class FacePreprocessor:
    """Enhanced face preprocessing with alignment and noise reduction"""

    def __init__(self, landmark_predictor_path = LANDMARK_PREDICTOR):
        self.predictor = dlib.shape_predictor(landmark_predictor_path)
        self.face_alignment_enabled = True

    def enhance_face_alignment(self, image, desired_face_width = 224, desired_left_eye=(0.35, 0.35)):
        """Enhanced face alignment with scaling and cropping"""

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        height, width = gray_img.shape[:2]
        face = dlib.rectangle(left = 0, top = 0, right = width - 1, bottom = height -1)
        shape = self.predictor(gray_img, face)

        # calculate eye centers from all eye landmarks
        left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

        left_eye_center = np.mean(left_eye_points, axis= 0)
        right_eye_center = np.mean(right_eye_points, axis= 0)

        # Calculate angle and distance
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        desired_right_eye_x = 1.0 - desired_left_eye[0]
        eyes_distance = np.sqrt((dx ** 2) + (dy ** 2))
        desired_distance = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width
        scale = desired_distance / eyes_distance

        # Calculate transformation
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                      (left_eye_center[1] + right_eye_center[1]) / 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        tX = desired_face_width * 0.5
        tY = desired_face_width * desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply transformation
        output_size = (desired_face_width, desired_face_width)
        aligned = cv2.warpAffine(image, M, output_size, flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def reduce_noise(self, image):
        """Apply noise reduction techniques"""
        # Bilateral filter - reduces noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def enhance_contrast(self, image):
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def normalize_illumination(self, image):
        """Normalize illumination variations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create illumination map using Gaussian blur
        illumination = cv2.GaussianBlur(gray, (0, 0), sigmaX=gray.shape[0]/30)
        
        # Normalize
        normalized = cv2.divide(gray, illumination, scale=255)
        
        if len(image.shape) == 3:
            # Apply normalization to all channels
            normalized_bgr = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                illumination_channel = cv2.GaussianBlur(channel, (0, 0), sigmaX=channel.shape[0]/30)
                normalized_bgr[:, :, i] = cv2.divide(channel, illumination_channel, scale=255)
            return normalized_bgr
        
        return normalized
    
    def preprocess_image(self, image_path_or_array, target_size=224):
        """Complete preprocessing pipeline"""
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array.copy()
        
        if image is None:
            raise ValueError("Could not load image")
        
        # 1. Face alignment (most important step)
        aligned = self.enhance_face_alignment(image, desired_face_width=target_size)
        
        # 2. Noise reduction
        denoised = self.reduce_noise(aligned)
        
        # 3. Contrast enhancement
        enhanced = self.enhance_contrast(denoised)
        
        # 4. Illumination normalization
        normalized = self.normalize_illumination(enhanced)
        
        # 5. Convert to RGB if needed
        if len(normalized.shape) == 3:
            final_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        else:
            # Convert grayscale to RGB by replicating channels
            final_image = np.stack([normalized] * 3, axis=-1)
        
        return final_image


class PreprocessedDataset(Dataset):
    """Custom dataset with enhanced preprocessing"""
    
    def __init__(self, root_dir, transform=None, preprocessor=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        
        # Apply preprocessing if available
        if self.preprocessor:
            try:
                image = self.preprocessor.preprocess_image(image_path)
                image = Image.fromarray(image.astype(np.uint8))
            except Exception as e:
                # Fallback to original loading
                image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



