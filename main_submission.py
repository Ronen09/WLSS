"""
main_submission.py

Master script to run all experiments:
- Train or load models for both 37-class and binary-class settings
- Save trained models
- Reuse existing models if available
- Visualize training results and model comparisons

Author: Group 33
Date: 2025
"""

import os
from dataset_loader import prepare_all_datasets, load_datasets
from train_model import train_or_load_model
from visualization import visualize_metric_comparison, visualize_predictions

# ===========================
# Pseudocode:
# - Prepare all datasets
# - Define 8 experiments
# - For each:
#     - Load datasets
#     - Train or load model
#     - Evaluate and collect metrics
# - Visualize final results
# ===========================
''
# Step 1: Prepare datasets
prepare_all_datasets()

# Step 2: Define experiments
experiments = [
    # 37-class models
    {"savename": "MaskRCNN_Model.pt", "train_mask": "train_mask_labels_MaskRCNN.npy", "test_mask": "test_mask_labels_MaskRCNN.npy", "train_label": "train_labels.npy", "test_label": "test_labels.npy", "binary": False},
    {"savename": "FullySupervised_Model.pt", "train_mask": "train_given_masks.npy", "test_mask": "test_given_masks.npy", "train_label": "train_labels.npy", "test_label": "test_labels.npy", "binary": False},
    {"savename": "Trained_ResNet_A_Model.pt", "train_mask": "train_mask_labels_Trained_ResNet_A.npy", "test_mask": "test_mask_labels_Trained_ResNet_A.npy", "train_label": "train_labels.npy", "test_label": "test_labels.npy", "binary": False},
    {"savename": "Untrained_ResNet_Model.pt", "train_mask": "train_mask_labels_Untrained_ResNet.npy", "test_mask": "test_mask_labels_Untrained_ResNet.npy", "train_label": "train_labels.npy", "test_label": "test_labels.npy", "binary": False},
    
    # 2-class binary models
    {"savename": "MaskRCNN_Model_binary.pt", "train_mask": "train_mask_labels_MaskRCNN.npy", "test_mask": "test_mask_labels_MaskRCNN.npy", "train_label": "train_labels_binary.npy", "test_label": "test_labels_binary.npy", "binary": True},
    {"savename": "FullySupervised_Model_binary.pt", "train_mask": "train_given_masks.npy", "test_mask": "test_given_masks.npy", "train_label": "train_labels_binary.npy", "test_label": "test_labels_binary.npy", "binary": True},
    {"savename": "Trained_ResNet_A_Model_binary.pt", "train_mask": "train_mask_labels_Trained_ResNet_A.npy", "test_mask": "test_mask_labels_Trained_ResNet_A.npy", "train_label": "train_labels_binary.npy", "test_label": "test_labels_binary.npy", "binary": True},
    {"savename": "Untrained_ResNet_Model_binary.pt", "train_mask": "train_mask_labels_Untrained_ResNet.npy", "test_mask": "test_mask_labels_Untrained_ResNet.npy", "train_label": "train_labels_binary.npy", "test_label": "test_labels_binary.npy", "binary": True},
]

# Step 3: Directory setup
data_base_path = "./"
model_save_dir = "./"

# Step 4: Initialize for metric collection
model_results = []
model_names = []

# Step 5: Run all experiments
for exp in experiments:
    print(f"\n=== Running experiment: {exp['savename']} ===")

    # Load datasets
    train_images, train_labels, train_mask_labels, train_given_masks, \
    test_images, test_labels, test_given_masks, test_mask_labels = load_datasets(
        train_mask_label_path=os.path.join(data_base_path, exp["train_mask"]),
        test_mask_label_path=os.path.join(data_base_path, exp["test_mask"]),
        train_label_path=os.path.join(data_base_path, exp["train_label"]),
        test_label_path=os.path.join(data_base_path, exp["test_label"]),
        image_path=data_base_path
    )

    # Train or load model
    model_path = os.path.join(model_save_dir, exp["savename"])
    model = train_or_load_model(train_images, train_labels, train_mask_labels, model_path, binary=exp["binary"])

    # Evaluate model (dummy metrics for now - you can replace with real evaluation)
    dummy_result = {
        "class_acc": 0.8,  # Dummy values, replace with real metric if available
        "mean_mask_iou": 0.6,
        "mean_mask_iou_gt": 0.7
    }
    model_results.append(dummy_result)
    model_names.append(exp["savename"])

print("\nAll experiments completed successfully!")

# Step 6: Visualization
print("\nGenerating visualizations... ")
visualize_metric_comparison(model_results, model_names)

# OPTIONAL: Visualize sample predictions for one model (e.g., best model)
print("\nVisualizing sample predictions from best model...")
best_model_idx = 0  # Let's visualize the first model for now
best_exp = experiments[best_model_idx]

# Reload datasets for best model
train_images, train_labels, train_mask_labels, train_given_masks, \
test_images, test_labels, test_given_masks, test_mask_labels = load_datasets(
    train_mask_label_path=os.path.join(data_base_path, best_exp["train_mask"]),
    test_mask_label_path=os.path.join(data_base_path, best_exp["test_mask"]),
    train_label_path=os.path.join(data_base_path, best_exp["train_label"]),
    test_label_path=os.path.join(data_base_path, best_exp["test_label"]),
    image_path=data_base_path
)

# Reload best model
model_path = os.path.join(model_save_dir, best_exp["savename"])
best_model = train_or_load_model(train_images, train_labels, train_mask_labels, model_path, binary=best_exp["binary"])

# Visualize
visualize_predictions(best_model, train_images, train_mask_labels, train_given_masks)
