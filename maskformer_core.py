"""
maskformer_core.py

This module contains:
- MaskFormer model definition
- Custom Dataset class for loading images, masks, labels
- Model training function
- Model evaluation functions
- Utility to set different learning rates for different model parts

Author: Group 33
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ===========================
# Pseudocode:
# - Define MaskFormer model (CNN encoder + Transformer decoder + Heads)
# - Define Dataset class
# - Define train_model function
# - Define evaluation functions
# ===========================

# --------------------------
# Model definition
# --------------------------

class MaskFormer(nn.Module):
    """
    MaskFormer model combining CNN encoder, transformer decoder,
    mask prediction head, and class prediction head.
    """

    def __init__(self, num_queries=37, num_classes=37, hidden_dim=128):
        super(MaskFormer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=2)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, images):
        x = self.encoder(images)
        bs, c, h, w = x.shape
        pos = torch.randn_like(x)  # Simple positional encoding
        hs = self.transformer_decoder(self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1), (x + pos).flatten(2).permute(2, 0, 1))
        outputs_class = self.class_embed(hs)
        mask_features = self.mask_embed(x)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", hs, mask_features)
        return outputs_class, outputs_mask

# --------------------------
# Custom Dataset class
# --------------------------

class ImageMaskDataset(Dataset):
    """
    PyTorch Dataset for loading images, masks, and labels.
    """

    def __init__(self, images, masks, labels):
        self.images = images
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).permute(2, 0, 1).float()
        mask = torch.tensor(self.masks[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return image, mask, label

# --------------------------
# Training function
# --------------------------

def train_model(model, train_loader, train_labels, optimizer, num_epochs=10, savename="model"):
    """
    Train MaskFormer model.

    Args:
        model (nn.Module): MaskFormer instance.
        train_loader (DataLoader): DataLoader for training.
        train_labels (np.ndarray): Labels array.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs.
        savename (str): Save path prefix.

    Returns:
        dict: Training history
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    history = {"loss": []}

    criterion_class = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs_class, outputs_mask = model(images)

            class_loss = criterion_class(outputs_class.mean(dim=0), labels)
            mask_loss = criterion_mask(outputs_mask, masks)

            loss = class_loss + mask_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        history["loss"].append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"{savename}.pt")
    print(f"Model saved as {savename}.pt")

    return history

# --------------------------
# Utility for optimizer
# --------------------------

def model_parameter_groups(model):
    """
    Split model parameters into groups for setting different learning rates.

    Args:
        model (nn.Module): MaskFormer instance.

    Returns:
        dict: Parameter groups
    """
    groups = {
        "CNN": list(model.encoder.parameters()),
        "transformer_decoder": list(model.transformer_decoder.parameters()),
        "class_head": list(model.class_embed.parameters()),
        "pixel_decoder": list(model.mask_embed.parameters()),
        "mask_head": list(model.mask_embed.parameters())
    }
    return groups

# --------------------------
# Evaluation function
# --------------------------

def specific_pseudomasks_evaluate_model(model, train_images, train_labels, train_mask_labels, train_given_masks,
                                        test_images, test_labels, test_mask_labels, test_given_masks):
    """
    Evaluate model performance on train and test datasets.

    Returns:
        Evaluation metrics
    """
    # Placeholder, for now only return dummy values
    return None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
