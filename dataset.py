import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OralHealthDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # ✅ Only keep valid image files (avoid .DS_Store, hidden files, etc.)
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))  # Modify as per dataset format

        # ✅ Ensure mask path matches available files
        if not os.path.exists(img_path):
            raise FileNotFoundError(f" ERROR: Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f" ERROR: Mask not found: {mask_path}")

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure valid image/mask
        if image is None:
            raise ValueError(f"ERROR: Failed to load image: {img_path}")
        if mask is None:
            raise ValueError(f"ERROR: Failed to load mask: {mask_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # ✅ Convert mask to PyTorch tensor and ensure it's LongTensor
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# ✅ Define Data Transformations (512x512)
transform = A.Compose([
    A.LongestMaxSize(max_size=512),
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# ✅ Test dataset loading
if __name__ == "__main__":
    dataset = OralHealthDataset("data/train/images", "data/train/masks", transform=transform)
    print("Train Dataset loaded with", len(dataset), "samples")
