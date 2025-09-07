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
def plot_data(data, fig_label, fig_title, type_of_fig, color = 'blue', x_label = 'Epoch', y_label = 'Value', root_path = './modules/pose_estimation/figures'):
    plt.figure(figsize = (8,5))
    plt.plot(data, label = fig_label, color = color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # save figure
    plt.savefig(os.path.join(root_path, type_of_fig))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names = None, save_path = './modules/pose_estimation/figures/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    # plt.title('Confusion Matrix', fontsize = 24, fontweight = 'bold', pad = 20, loc = 'center')

    plt.xlabel('Predicted labels', fontsize=15, fontweight='bold', labelpad=20)
    plt.ylabel('True labels', fontsize=15, fontweight='bold', labelpad=20)

    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")

def plot_metrics_report(y_true, y_pred, class_names, save_path='./modules/pose_estimation/figures/classification_metrics_bar.png'):
    from sklearn.metrics import precision_recall_fscore_support
    import numpy as np
    import matplotlib.pyplot as plt

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width, precision, width, label='Precision')
    bars2 = plt.bar(x, recall, width, label='Recall')
    bars3 = plt.bar(x + width, f1, width, label='F1-Score')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score per Class')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()

    # Thêm text lên từng bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,  # vị trí text ngay trên bar
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Metrics bar chart saved to {save_path}")