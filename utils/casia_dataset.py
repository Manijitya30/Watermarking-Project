import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class CASIADataset(Dataset):
    def __init__(self, root_dir, transform=None, use_handcrafted=False, target_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.use_handcrafted = use_handcrafted
        self.target_size = target_size

        self.images = []
        self.labels = []

        au_dir = os.path.join(root_dir, "Au")
        tp_dir = os.path.join(root_dir, "Tp")

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        # Authentic = 0
        for img in os.listdir(au_dir):
            path = os.path.join(au_dir, img)
            if img.lower().endswith(valid_ext):
                self.images.append(path)
                self.labels.append(0)

        # Forged = 1
        for img in os.listdir(tp_dir):
            path = os.path.join(tp_dir, img)
            if img.lower().endswith(valid_ext):
                self.images.append(path)
                self.labels.append(1)

        print(f"[CASIA] {len(self.images)} images loaded")
        print(f"   Authentic: {self.labels.count(0)} | Forged: {self.labels.count(1)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Resize((self.target_size, self.target_size))(image)
            image = transforms.ToTensor()(image)

        # return same format as CoMoFoD
        if self.use_handcrafted:
            return image, torch.zeros(39), label

        return image, label