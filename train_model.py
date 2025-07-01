"""
train_model.py

This module handles training and loading MaskFormer models:
- Checks if a saved model exists.
- If yes, loads the model.
- If no, trains a new model and saves it.

Author: Group 33
Date: 2025
"""

import torch
import os
from torch.utils.data import DataLoader
from maskformer_core import MaskFormer, model_parameter_groups, train_model, ImageMaskDataset

# ===========================
# Pseudocode:
# - If model .pt exists → load model
# - If not exist → train model and save
# - Handles both 37-class and 2-class cases
# ===========================

def train_or_load_model(train_images, train_labels, train_mask_labels, model_path, binary=False):
    """
    Train a new MaskFormer model if no saved model exists, 
    otherwise load the existing model.

    Args:
        train_images (np.ndarray): Training images.
        train_labels (np.ndarray): Training class labels (37 or 2 classes).
        train_mask_labels (np.ndarray): Training pseudo-masks.
        model_path (str): Path to save/load the model (.pt).
        binary (bool): If True, configure the model for 2-class binary classification.

    Returns:
        torch.nn.Module: Trained or loaded MaskFormer model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If model already exists, load it
    if os.path.exists(model_path):
        print(f"Model {model_path} already exists. Loading... ✅")
        model = MaskFormer(
            num_queries=2 if binary else 37,
            num_classes=2 if binary else 37,
            hidden_dim=128
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    # Otherwise, train a new model
    print(f"Training new model: {model_path} 🚀")
    model = MaskFormer(
        num_queries=2 if binary else 37,
        num_classes=2 if binary else 37,
        hidden_dim=128
    ).to(device)

    # Define optimizer with different learning rates
    # Define optimizer with different learning rates
    parameter_groups = model_parameter_groups(model)
    optimizer = torch.optim.AdamW([
        {'params': parameter_groups['CNN'], 'lr': 1e-5, 'weight_decay': 1e-5},
        {'params': parameter_groups['transformer_decoder'], 'lr': 1e-5, 'weight_decay': 1e-5},
        {'params': parameter_groups['class_head'], 'lr': 1e-4, 'weight_decay': 1e-4},
        {'params': parameter_groups['mask_head'], 'lr': 5e-5, 'weight_decay': 5e-5},
    ], lr=1e-5)


    # Prepare DataLoader
    train_dataloader = DataLoader(
        ImageMaskDataset(train_images, train_mask_labels, train_labels),
        batch_size=32,
        shuffle=True
    )

    # Train the model
    history = train_model(
        model,
        train_dataloader,
        train_labels,
        optimizer,
        num_epochs=10,
        savename=model_path[:-3]  # Remove '.pt' from filename
    )

    return model
