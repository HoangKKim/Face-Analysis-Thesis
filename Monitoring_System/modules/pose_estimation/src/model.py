import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from collections import defaultdict

from cfg.pose_cfg import *
from modules.pose_estimation.src.visualize import *
from tqdm import tqdm

class BehaviorClassifier(nn.Module):
    def __init__(self, input_size = 36, num_classes = 5):
        super().__init__()
        # add drop out to reduce percentage of overfitting
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def load_data(feature_path, type_of_features):
    X = np.load(os.path.join(feature_path, type_of_features + "_features.npy"))
    y = np.load(os.path.join(feature_path, type_of_features + "_labels.npy"))

    # preprocess
    if type_of_features == 'train':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # save scaler after fitting
        joblib.dump(scaler, BEHAVIOR_SCALER_PATH)
    else:
        if not os.path.exists(BEHAVIOR_SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {BEHAVIOR_SCALER_PATH}. You must run training first.")
        scaler = joblib.load(BEHAVIOR_SCALER_PATH)
        X = scaler.transform(X)
    return X, y

def train_model_single_fold(model, train_loader, valid_loader, device, num_epochs=50, lr=5e-4, patience=5):
    """Train model for a single fold - returns best validation accuracy"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_val_acc = 0.0
    epochs_no_improve = 0

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0 

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        correct = total = 0
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_accuracy = correct / total * 100
        
        # Early stopping based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_val_acc

def train_model_kfold(X_train, y_train, input_size, num_classes, device, k_folds=5, num_epochs=50, lr=5e-4, patience=5):
    """
    Perform K-Fold Cross Validation
    Returns: list of fold accuracies, mean accuracy, std accuracy
    """
    print(f"\n🔄 Starting {k_folds}-Fold Cross Validation")
    print("=" * 60)
    
    # Initialize K-Fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_results = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\n📁 Training Fold {fold + 1}/{k_folds}")
        print("-" * 40)
        
        # Split data for current fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Scale data for current fold
        fold_scaler = StandardScaler()
        X_fold_train = fold_scaler.fit_transform(X_fold_train)
        X_fold_val = fold_scaler.transform(X_fold_val)
        
        # Create datasets and loaders
        fold_train_dataset = FeatureDataset(X_fold_train, y_fold_train)
        fold_val_dataset = FeatureDataset(X_fold_val, y_fold_val)
        
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=16, shuffle=True, drop_last = True)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=16, shuffle=False)
        
        # Initialize fresh model for each fold
        fold_model = BehaviorClassifier(input_size=input_size, num_classes=num_classes)
        
        # Train model for current fold
        fold_accuracy = train_model_single_fold(
            fold_model, fold_train_loader, fold_val_loader, 
            device, num_epochs, lr, patience
        )
        
        fold_accuracies.append(fold_accuracy)
        print(f"✅ Fold {fold + 1} Validation Accuracy: {fold_accuracy:.2f}%")
        
    # Calculate statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print("\n" + "=" * 60)
    print("📊 K-FOLD CROSS VALIDATION RESULTS")
    print("=" * 60)
    print(f"Individual Fold Accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    print(f"Mean CV Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
    print(f"Best Fold Accuracy: {max(fold_accuracies):.2f}%")
    print(f"Worst Fold Accuracy: {min(fold_accuracies):.2f}%")
    print("=" * 60)
    
    return fold_accuracies, mean_accuracy, std_accuracy

def train_model_final(model, train_loader, valid_loader, device, num_epochs=50, lr=5e-4, ckpt_path='best_checkpoint.pth', patience=5):
    """Original training function for final model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_val_loss = float('inf')
    best_epoch = -1
    best_ckpt_path = None
    epochs_no_improve = 0

    model.to(device)

    loss_history = []
    accuracy_history = []

    for epoch in tqdm(range(num_epochs), desc="🚀 Final Training Progress", unit="epoch"):
        model.train()
        total_loss = 0 

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = total = 0
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_val_loss = val_loss / len(valid_loader)
        accuracy = correct / total * 100

        # Save history
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, ckpt_path)
            best_ckpt_path = ckpt_path
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_ckpt_path:
        print(f"\nBest final model saved at epoch {best_epoch} ; Path: '{best_ckpt_path}'")
    else:
        print("\nNo model was saved during final training.")

    return loss_history, accuracy_history

def test_model(model, checkpoint_path, test_loader, device='cuda', class_names = CLASS_NAMES, report_path= 'modules/pose_estimation/figures/classification_report.txt'):
    from sklearn.metrics import classification_report

    # load pretrained checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # init pretrained model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    correct = total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    plot_confusion_matrix(all_labels, all_preds, class_names=class_names)

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Save report
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] Classification report saved to {report_path}")

    plot_metrics_report(all_labels, all_preds, class_names)

    return accuracy

def plot_kfold_results(fold_accuracies, mean_accuracy, std_accuracy, save_path='modules/pose_estimation/figures'):
    """Plot K-Fold Cross Validation Results"""
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Bar chart of fold accuracies
    plt.subplot(1, 2, 1)
    folds = [f'Fold {i+1}' for i in range(len(fold_accuracies))]
    bars = plt.bar(folds, fold_accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axhline(y=mean_accuracy, color='red', linestyle='--', label=f'Mean: {mean_accuracy:.2f}%')
    plt.axhline(y=mean_accuracy + std_accuracy, color='orange', linestyle=':', alpha=0.7, label=f'+1 STD: {mean_accuracy + std_accuracy:.2f}%')
    plt.axhline(y=mean_accuracy - std_accuracy, color='orange', linestyle=':', alpha=0.7, label=f'-1 STD: {mean_accuracy - std_accuracy:.2f}%')
    
    # Add value labels on bars
    for bar, acc in zip(bars, fold_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('K-Fold Cross Validation Results', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(fold_accuracies, labels=['All Folds'])
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Accuracy Distribution Across Folds', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {mean_accuracy:.2f}%\nStd: {std_accuracy:.2f}%\nRange: {max(fold_accuracies) - min(fold_accuracies):.2f}%'
    plt.text(1.15, min(fold_accuracies), stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    kfold_plot_path = os.path.join(save_path, 'kfold_results.png')
    plt.savefig(kfold_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"[INFO] K-Fold results plot saved to {kfold_plot_path}")

if __name__ == '__main__':
    # ------------------------------------------- Step 1 - Load Data ------------------------------------------------------
    print(" Loading data...")
    
    # For K-Fold, we combine train and validation data
    X_train_orig, y_train_orig = load_data(BEHAVIOR_FEATURE_DIR, 'train')
    X_valid_orig, y_valid_orig = load_data(BEHAVIOR_FEATURE_DIR, 'valid') 
    X_test, y_test = load_data(BEHAVIOR_FEATURE_DIR, 'test') 

    # Combine train and validation for K-Fold CV
    X_combined = np.vstack([X_train_orig, X_valid_orig])
    y_combined = np.hstack([y_train_orig, y_valid_orig])
    
    print(f"Combined dataset shape: {X_combined.shape}")
    print(f"Test dataset shape: {X_test.shape}")

    # Get number of classes automatically 
    num_classes = len(np.unique(y_combined))
    input_size = X_combined.shape[1]
    
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for training')

    # ------------------------------------------- Step 2 - K-Fold Cross Validation ------------------------------------------------------
    
    # Perform K-Fold Cross Validation
    fold_accuracies, mean_cv_accuracy, std_cv_accuracy = train_model_kfold(
        X_combined, y_combined, input_size, num_classes, device, 
        k_folds=10, num_epochs=100, lr=5e-4, patience=10
    )
    
    # Plot K-Fold results
    plot_kfold_results(fold_accuracies, mean_cv_accuracy, std_cv_accuracy)
    
    # ------------------------------------------- Step 3 - Train Final Model ------------------------------------------------------
    
    print("\n🎯 Training final model on full dataset...")
    
    # Use original train/validation split for final model training
    train_dataset = FeatureDataset(X_train_orig, y_train_orig)
    valid_dataset = FeatureDataset(X_valid_orig, y_valid_orig)
    test_dataset = FeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize final model 
    final_model = BehaviorClassifier(input_size=input_size, num_classes=num_classes)

    # Train final model
    loss, accuracy = train_model_final(final_model, train_loader, valid_loader, device, 
                                     num_epochs=100, ckpt_path=BEHAVIOR_CKPT_PATH, patience=10)

    # Plot training curves for final model and save to figures directory
    save_path = 'modules/pose_estimation/figures'
    os.makedirs(save_path, exist_ok=True)
    
    # Save training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss, color='blue', linewidth=2, label='Training Loss')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    train_loss_path = os.path.join(save_path, 'train_loss.png')
    plt.savefig(train_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training loss plot saved to {train_loss_path}")
    
    # Save validation accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, color='green', linewidth=2, label='Validation Accuracy')
    plt.title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    valid_acc_path = os.path.join(save_path, 'valid_accuracy.png')
    plt.savefig(valid_acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Validation accuracy plot saved to {valid_acc_path}")

    # ------------------------------------------- Step 4 - Test Final Model ------------------------------------------------------
    
    print(f"\n📊 Final Results Summary:")
    print(f"Cross-Validation Accuracy: {mean_cv_accuracy:.2f} ± {std_cv_accuracy:.2f}%")
    
    # Test final model
    final_test_accuracy = test_model(final_model, BEHAVIOR_CKPT_PATH, test_loader, device=device)
    
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
    print(f"CV vs Test Difference: {abs(mean_cv_accuracy - final_test_accuracy):.2f}%")
    
    if abs(mean_cv_accuracy - final_test_accuracy) < 3.0:
        print("✅ Good agreement between CV and test accuracy!")
    else:
        print("⚠️  Large difference between CV and test accuracy - possible overfitting or data leakage")