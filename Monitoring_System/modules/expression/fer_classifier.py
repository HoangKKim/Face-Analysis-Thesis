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

from modules.expression.preprocessor import * 



from cfg.expression_cfg import * 

class FERClassifier:
    def __init__(self,
                 root_dir,
                 batch_size=16,
                 learning_rate=5e-5,
                 weight_decay=1e-3,
                 patience=3,
                 num_epochs=30,
                 num_classes=7,
                 weights_path=None,
                 predictor_path=LANDMARK_PREDICTOR,
                 use_advanced_preprocessing=True):
        
        # Setup parameters
        self.device = DEVICE
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.num_epochs = num_epochs
        self.root_dir = root_dir
        self.train_path = os.path.join(root_dir, 'train')
        self.test_path = os.path.join(root_dir, 'test')
        
        # Initialize preprocessor
        self.preprocessor = FacePreprocessor(predictor_path) if use_advanced_preprocessing else None
        
        # Initialize model
        self._setup_model(weights_path)
        
        # Training components
        self._setup_training_components()
        
        print(f"🚀 FER Classifier initialized on {self.device}")
        print(f"📊 Classes: {self.num_classes}, Batch size: {self.batch_size}")
        print(f"🔧 Advanced preprocessing: {'✅' if use_advanced_preprocessing else '❌'}")
    
    def _setup_model(self, weights_path):
        """Setup model architecture with improved classifier head"""
        # Load pretrained EfficientNet-B0
        base_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=base_weights)
        
        # Enhanced classifier head with better regularization
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Higher dropout before final layer
            nn.Linear(256, self.num_classes)
        )
        
        # Load fine-tuned weights if provided
        if weights_path and os.path.isfile(weights_path):
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ Loaded fine-tuned weights from: {weights_path}")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")
        else:
            print("🔄 Using ImageNet pretrained weights only")
        
        self.model = self.model.to(self.device)
    
    def _setup_training_components(self):
        """Setup loss, optimizer, and scheduler"""
        # Use label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use AdamW with proper weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Improved scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping
        self.best_acc = 0.0
        self.epochs_no_improve = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
    
    def get_advanced_transforms(self):
        """Get advanced data augmentation transforms"""
        """SOLUTION 5: More aggressive data augmentation"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def load_data(self):
        """Load and prepare datasets with enhanced preprocessing"""
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Train or test path does not exist: {self.train_path}, {self.test_path}")
        
        train_transform, test_transform = self.get_advanced_transforms()
        
        # Create datasets with preprocessing
        full_dataset = PreprocessedDataset(
            self.train_path, 
            transform=train_transform, 
            preprocessor=self.preprocessor
        )
        
        # Split dataset
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create validation dataset with test transforms
        val_dataset_transformed = PreprocessedDataset(
            self.train_path, 
            transform=test_transform, 
            preprocessor=self.preprocessor
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        test_dataset = PreprocessedDataset(
            self.test_path, 
            transform=test_transform, 
            preprocessor=self.preprocessor
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        print(f"📊 Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    def train(self, model_dir):
        """Enhanced training loop with better monitoring"""
        os.makedirs(model_dir, exist_ok=True)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.num_epochs):
            print(f'\n🔄 Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 50)
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 49:
                    print(f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                          f"Loss: {running_loss/(batch_idx+1):.4f} | "
                          f"Acc: {100*correct_train/total_train:.2f}%")
            
            # Validation phase
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            epoch_train_loss = running_loss / len(self.train_loader)
            train_acc = 100 * correct_train / total_train
            
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"\n📈 Results:")
            print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping and model saving
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                
                model_path = os.path.join(model_dir, 'best_fer_model.pth')
                self.save_model(model_path)
                print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.epochs_no_improve += 1
                print(f"⚠️ No improvement for {self.epochs_no_improve} epoch(s)")
            
            if self.epochs_no_improve >= self.patience:
                print("⛔ Early stopping triggered!")
                break
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, val_accuracies, model_dir)
        
        # Load best model
        self.model.load_state_dict(self.best_model_wts)
        print(f"\n🎯 Training completed! Best validation accuracy: {self.best_acc:.2f}%")
    
    def evaluate(self, data_loader, return_predictions=False):
        """Enhanced evaluation with optional detailed metrics"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if return_predictions:
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        if return_predictions:
            return avg_loss, accuracy, all_predictions, all_labels
        return avg_loss, accuracy
    
    def test_model(self, class_names=None):
        """Comprehensive model testing with detailed metrics"""
        print("\n🧪 Testing model...")
        test_loss, test_acc, predictions, labels = self.evaluate(self.test_loader, return_predictions=True)
        
        print(f"📊 Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(labels, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return test_loss, test_acc
    
    def inference(self, image_path_or_array, return_probabilities=True):
        """Enhanced inference with preprocessing"""
        self.model.eval()
        
        # Preprocess image
        if self.preprocessor:
            processed_image = self.preprocessor.preprocess_image(image_path_or_array)
            image_pil = Image.fromarray(processed_image.astype(np.uint8))
        else:
            if isinstance(image_path_or_array, str):
                image_pil = Image.open(image_path_or_array).convert('RGB')
            else:
                image_pil = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))
        
        # Apply test transforms
        _, test_transform = self.get_advanced_transforms()
        image_tensor = test_transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted_class = predicted.cpu().numpy()[0]
        probs = probabilities.cpu().numpy()[0]
        
        if return_probabilities:
            return predicted_class, probs
        return predicted_class
    
    def plot_training_history(self, train_losses, val_losses, val_accuracies, save_dir):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path):
        """Save model with metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'best_acc': self.best_acc,
            'model_architecture': 'EfficientNet-B0'
        }, path)
    
    def load_model(self, path):
        """Load model with metadata"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        if 'best_acc' in checkpoint:
            self.best_acc = checkpoint['best_acc']
            print(f"✅ Model loaded with best accuracy: {self.best_acc:.2f}%")

if __name__ == "__main__":
    # Initialize classifier
    classifier = FERClassifier(
        root_dir="input/affectnet-dataset",
        use_advanced_preprocessing=True
    )

    classifier.load_model(FER_MODEL)
    # train & test
    # Load data
    # classifier.load_data()
    
    # # Train model
    # classifier.train("modules/expression/models")
    
    # Test model
    class_names = FER_LABEL
    # classifier.test_model(class_names)

    # inference
    img_dir = 'input/expression'
    for file in os.listdir(img_dir):
        predicted_class, probabilities = classifier.inference(os.path.join(img_dir, file))
        print(f"Predicted: {class_names[predicted_class]}")
        print(f"Probabilities: {probabilities}")
        print(f"Ground truth: {file}") 