"""
dataset_loader.py

Handles full dataset preparation:
- Checks if all .npy files exist.
- If missing, automatically downloads and processes Oxford-IIIT Pet Dataset.
- Generates pseudo-masks using Mask-RCNN if missing.
- Handles binary label generation for 2-class training.

Author: Group 33
Date: 2025
"""

import os
import numpy as np
from torchvision import datasets, transforms
import torch
import torchvision.models.segmentation as segmentation
import torchvision.transforms.functional as TF



# Define where data will be downloaded
DATA_FOLDER = os.path.join(os.getcwd(), "data")

def generate_images_and_labels():
    if all([
        os.path.exists("train_images.npy"),
        os.path.exists("test_images.npy"),
        os.path.exists("train_labels.npy"),
        os.path.exists("test_labels.npy")
    ]):
        print("Train/Test images and labels already exist. Skipping download.")
        return

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER, exist_ok=True)

    print("Downloading Oxford-IIIT Pet dataset...")

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_dataset = datasets.OxfordIIITPet(root=DATA_FOLDER, download=True, transform=transform)
    test_dataset = datasets.OxfordIIITPet(root=DATA_FOLDER, download=False, split="test", transform=transform)


    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img.permute(1,2,0).numpy())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.permute(1,2,0).numpy())
        test_labels.append(label)

    np.save("train_images.npy", np.array(train_images))
    np.save("test_images.npy", np.array(test_images))
    np.save("train_labels.npy", np.array(train_labels))
    np.save("test_labels.npy", np.array(test_labels))

    print("Train/Test images and labels saved successfully")


def generate_given_masks():
    """
    Generate dummy ground-truth masks if missing.
    """
    if os.path.exists("train_given_masks.npy") and os.path.exists("test_given_masks.npy"):
        print("Given masks already exist. Skipping generation.")
        return

    print("Generating dummy given masks...")
    train_images = np.load("train_images.npy")
    test_images = np.load("test_images.npy")

    train_masks = np.random.randint(0, 2, size=(len(train_images), 256, 256))
    test_masks = np.random.randint(0, 2, size=(len(test_images), 256, 256))

    np.save("train_given_masks.npy", train_masks)
    np.save("test_given_masks.npy", test_masks)

    print("Dummy given masks generated and saved")

def generate_pseudo_masks_MaskRCNN():
    """
    Generate pseudo masks using MaskRCNN model if missing.
    """
    if os.path.exists("train_mask_labels_MaskRCNN.npy") and os.path.exists("test_mask_labels_MaskRCNN.npy"):
        print("MaskRCNN pseudo-masks already exist. Skipping generation.")
        return

    print("Generating pseudo-masks with MaskRCNN...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = segmentation.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_images = np.load("train_images.npy")
    test_images = np.load("test_images.npy")

    def get_masks(images):
        masks = []
        with torch.no_grad():
            for img in images:
                img_tensor = transform((img*255).astype(np.uint8)).unsqueeze(0).to(device)
                outputs = model(img_tensor)[0]
                if len(outputs["masks"]) > 0:
                    mask = outputs["masks"][0, 0].cpu().numpy()
                    mask = (mask > 0.5).astype(np.uint8)
                else:
                    mask = np.zeros((256,256), dtype=np.uint8)
                masks.append(mask)
        return np.array(masks)

    train_masks = get_masks(train_images)
    test_masks = get_masks(test_images)

    np.save("train_mask_labels_MaskRCNN.npy", train_masks)
    np.save("test_mask_labels_MaskRCNN.npy", test_masks)

    print("MaskRCNN pseudo-masks generated and saved")

def generate_binary_labels():
    """
    Generate binary labels for 2-class classification (cat=1, dog=0).
    """
    if os.path.exists("train_labels_binary.npy") and os.path.exists("test_labels_binary.npy"):
        print("Binary labels already exist. Skipping generation.")
        return

    print("Generating binary labels...")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    train_dataset = datasets.OxfordIIITPet(root="./", download=True, transform=transform, target_types="binary-category")
    test_dataset = datasets.OxfordIIITPet(root="./", download=False, split="test", transform=transform, target_types="binary-category")

    binary_train_labels = [np.array(train_dataset[i][1]) for i in range(len(train_dataset))]
    binary_test_labels = [np.array(test_dataset[i][1]) for i in range(len(test_dataset))]

    np.save("train_labels_binary.npy", np.array(binary_train_labels))
    np.save("test_labels_binary.npy", np.array(binary_test_labels))

    print("Binary labels generated and saved ")

def prepare_all_datasets():
    """
    Prepare all datasets required for experiments.
    """
    print("Preparing full datasets... ")
    generate_images_and_labels()
    generate_given_masks()
    generate_pseudo_masks_MaskRCNN()
    generate_binary_labels()
    print("All datasets are ready! ")

def load_datasets(train_mask_label_path, test_mask_label_path, train_label_path, test_label_path, image_path="./"):
    """
    Load all required datasets for training and evaluation.

    Returns:
        Tuple of (train_images, train_labels, train_mask_labels, train_given_masks,
                  test_images, test_labels, test_given_masks, test_mask_labels)
    """

    prepare_all_datasets()

    print("Loading datasets from disk...")
    train_images = np.load(os.path.join(image_path, "train_images.npy"))
    test_images = np.load(os.path.join(image_path, "test_images.npy"))
    train_labels = np.load(train_label_path)
    test_labels = np.load(test_label_path)
    train_mask_labels = np.load(train_mask_label_path)
    test_mask_labels = np.load(test_mask_label_path)
    train_given_masks = np.load(os.path.join(image_path, "train_given_masks.npy"))
    test_given_masks = np.load(os.path.join(image_path, "test_given_masks.npy"))

    print("Datasets loaded successfully! ")
    return train_images, train_labels, train_mask_labels, train_given_masks, test_images, test_labels, test_given_masks, test_mask_labels
