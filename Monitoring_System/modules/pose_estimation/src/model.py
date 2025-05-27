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

from cfg.pose_cfg import *
from modules.pose_estimation.src.visualize import *

class BehaviorClassifier(nn.Module):
    def __init__(self, input_size = 26, num_classes = 5):
        super().__init__()
        # add drop out to reduce percentage of overfitting
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 16),
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

def train_model(model, train_loader, valid_loader, device, num_epochs=50, lr=1e-4, ckpt_path = 'best_checkpoint.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    model.to(device)
    
    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
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

        # Evaluation
        model.eval() 
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total * 100

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {accuracy:.2f}%")

        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_loss
            }, ckpt_path)
            print(f"[INFO] Saved best model at epoch {epoch} with val_loss = {avg_loss:.4f}")

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


if __name__ == '__main__':
    # ------------------------------------------- Step 1 - Train model ------------------------------------------------------

    # # load data
    # X_train, y_train = load_data(BEHAVIOR_FEATURE_DIR, 'train')
    # X_valid, y_valid = load_data(BEHAVIOR_FEATURE_DIR, 'valid') 
    # X_test, y_test = load_data(BEHAVIOR_FEATURE_DIR, 'test') 

    # # get number of class automatically 
    # num_classes = len(np.unique(y_train))

    # # dataset and dataloader
    # train_dataset = FeatureDataset(X_train, y_train)
    # valid_dataset = FeatureDataset(X_valid, y_valid)
    # test_dataset = FeatureDataset(X_test, y_test)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=32)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # # init model 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using {device} for training')

    # model = BehaviorClassifier(input_size= 26, num_classes= num_classes)

    # # train model
    # loss, accuracy = train_model(model, train_loader, valid_loader, device, 200, ckpt_path=BEHAVIOR_CKPT_PATH)

    # # plot training data
    # plot_data(loss, "Training Loss", "Training Loss Curve", "train_loss", "blue", "Epochs", "Loss", )
    # plot_data(accuracy, "Validation Accuracy", "Validation Accuracy Curve", "valid_accuracy", "green", "Epochs", "Accuracy")

    # num_classes = len(np.unique(y_test))

    # # test model
    # test_model(model, BEHAVIOR_CKPT_PATH, test_loader, device='cuda')

    # ------------------------------------------- Step 22 - Test model ------------------------------------------------------
    X_test, y_test = load_data(BEHAVIOR_FEATURE_DIR, 'test') 
    num_classes = len(np.unique(y_test))
    test_dataset = FeatureDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BehaviorClassifier(input_size= 26, num_classes= num_classes)
    test_model(model, BEHAVIOR_CKPT_PATH, test_loader, device='cuda')




