"""
visualization_tools.py

This module provides functions to visualize:
- Training loss curves
- Model predictions
- Metric comparisons across models
- Mask comparisons across models

Author: Group 33
Date: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap

# ===========================
# 1. Loss Visualization
# ===========================

def plot_losses(total_losses, class_losses, mask_losses, num_epochs):
    """
    Plot total, class, and mask losses over training epochs.
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, label='Total Loss')
    plt.plot(epochs, class_losses, label='Class Loss')
    plt.plot(epochs, mask_losses, label='Mask Loss')
    plt.title('Training Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_losses.png')
    plt.show()

# ===========================
# 2. Prediction Visualization
# ===========================

def visualize_predictions(model, images, true_masks, given_masks, class_labels=None,
                          num_samples=5, save_dir='visualization_results',
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Visualize model predictions compared to pseudo and ground truth masks.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    model.to(device)

    indices = np.random.choice(len(images), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image = torch.tensor(images[idx], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            class_prediction, mask_prediction = model(image)
            class_pred = class_prediction[0].diagonal(dim1=0, dim2=1)
            class_pred = torch.argmax(torch.softmax(class_pred, dim=-1))
            mask_pred = (torch.sigmoid(mask_prediction[0, int(class_pred)]) > 0.5).cpu().numpy()

        img = images[idx].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(std * img + mean, 0, 1)

        pseudo_mask = true_masks[idx].astype(np.float32)
        gt_mask = given_masks[idx].astype(np.float32)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {i+1}/{num_samples} (Class: {class_labels[idx] if class_labels is not None else "N/A"})')

        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img)
        axes[0, 1].imshow(np.ma.masked_where(mask_pred == 0, mask_pred), alpha=0.5, cmap='cool')
        axes[0, 1].set_title("Predicted Mask Overlay")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(img)
        axes[0, 2].imshow(np.ma.masked_where(pseudo_mask == 0, pseudo_mask), alpha=0.5, cmap='autumn')
        axes[0, 2].set_title("Pseudo-Mask Overlay")
        axes[0, 2].axis('off')

        axes[1, 0].imshow(img)
        axes[1, 0].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), alpha=0.5, cmap='summer')
        axes[1, 0].set_title("Ground Truth Mask Overlay")
        axes[1, 0].axis('off')

        comparison = np.zeros_like(mask_pred, dtype=np.uint8)
        comparison[(mask_pred == 1) & (gt_mask == 1)] = 1
        comparison[(mask_pred == 1) & (gt_mask == 0)] = 2
        comparison[(mask_pred == 0) & (gt_mask == 1)] = 3
        cmap = ListedColormap(['black', 'green', 'red', 'blue'])

        axes[1, 1].imshow(comparison, cmap=cmap)
        axes[1, 1].set_title("Prediction vs Ground Truth")
        axes[1, 1].axis('off')

        comparison_pseudo = np.zeros_like(mask_pred, dtype=np.uint8)
        comparison_pseudo[(mask_pred == 1) & (pseudo_mask == 1)] = 1
        comparison_pseudo[(mask_pred == 1) & (pseudo_mask == 0)] = 2
        comparison_pseudo[(mask_pred == 0) & (pseudo_mask == 1)] = 3

        axes[1, 2].imshow(comparison_pseudo, cmap=cmap)
        axes[1, 2].set_title("Prediction vs Pseudo-Mask")
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
        plt.close()

# ===========================
# 3. Metric Comparison Visualization
# ===========================

def visualize_metric_comparison(model_results, model_names, save_dir='visualization_results'):
    """
    Plot bar charts comparing classification accuracy and mask IoU across models.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    class_accs = [r['class_acc'] for r in model_results]
    mask_ious = [r['mean_mask_iou'] for r in model_results]
    mask_ious_gt = [r['mean_mask_iou_gt'] for r in model_results]

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.25
    x = np.arange(len(model_names))

    ax.bar(x - bar_width, class_accs, bar_width, label='Class Accuracy')
    ax.bar(x, mask_ious, bar_width, label='Mask IoU (Pseudo)')
    ax.bar(x + bar_width, mask_ious_gt, bar_width, label='Mask IoU (Ground Truth)')

    ax.set_xlabel('Models')
    ax.set_ylabel('Metrics')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()
