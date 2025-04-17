import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from unet_model import UNet  # Import U-Net model
from dataset import transform  # Use the same preprocessing transformations

# ✅ Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=5)  # 5 classes including background
model.load_state_dict(torch.load("unet_model_512.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# ✅ Define Test Dataset Path
test_images_dir = "/teamspace/studios/this_studio/UNet/test/images"
test_masks_dir = "/teamspace/studios/this_studio/UNet/test/masks"  # Ground truth masks directory

# ✅ Evaluation Metrics Storage
dice_scores = []
iou_scores = []
precisions = []
recalls = []

# ✅ Function to Compute Dice Score
def dice_coefficient(pred, target, num_classes=5):
    dice = []
    for class_id in range(1, num_classes):  # Ignore background (class 0)
        pred_class = (pred == class_id).astype(np.uint8)
        target_class = (target == class_id).astype(np.uint8)

        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class)

        dice_score = (2. * intersection) / (union + 1e-6) if union > 0 else 1.0
        dice.append(dice_score)

    return np.mean(dice)

# ✅ Function to Compute Evaluation Metrics
def evaluate_model():
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("❌ No test images found.")
        return

    for image_name in image_files:
        image_path = os.path.join(test_images_dir, image_name)
    
        # ✅ Ensure correct mask file extension
        mask_path = os.path.join(test_masks_dir, os.path.splitext(image_name)[0] + ".png")

        if not os.path.exists(mask_path):
            print(f"❌ ERROR: Ground truth mask not found for {image_name}")
            continue

        # ✅ Load Image & Mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"❌ ERROR: Could not load {image_name} or its mask")
            continue

        original_size = (image.shape[1], image.shape[0])  # Save original size (width, height)

        # ✅ Resize image to 512x512 before prediction
        image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Apply same preprocessing as training
        transformed = transform(image=image_resized)
        image_tensor = transformed["image"].unsqueeze(0).to(device)

        # ✅ Forward pass through the model
        with torch.no_grad():
            output = model(image_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get class with max probability

        # ✅ Resize predicted mask back to original size
        mask_pred_resized = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        # ✅ Compute Dice Score
        dice = dice_coefficient(mask_pred_resized, mask)
        dice_scores.append(dice)

        # ✅ Compute IoU (Jaccard Index)
        iou = jaccard_score(mask.flatten(), mask_pred_resized.flatten(), average="macro")
        iou_scores.append(iou)

        # ✅ Compute Precision & Recall
        precision = precision_score(mask.flatten(), mask_pred_resized.flatten(), average="macro", zero_division=1)
        recall = recall_score(mask.flatten(), mask_pred_resized.flatten(), average="macro", zero_division=1)

        precisions.append(precision)
        recalls.append(recall)

        # print(f"✅ Evaluated {image_name} | Dice: {dice:.4f} | IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # ✅ Print Final Average Metrics
    print("\n🎯 Final Evaluation Results:")
    print(f"🔥 Average Dice Score: {np.mean(dice_scores):.4f}")
    print(f"🔥 Average IoU Score: {np.mean(iou_scores):.4f}")
    print(f"🔥 Average Precision: {np.mean(precisions):.4f}")
    print(f"🔥 Average Recall: {np.mean(recalls):.4f}")

# ✅ Run Evaluation
evaluate_model()
