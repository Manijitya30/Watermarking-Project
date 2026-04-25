import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from .handcrafted import extract_dct_features, extract_zernike_features, extract_lbp_features

class CoMoFoDOptimizedDataset(Dataset):

    def __init__(self, root_dir, transform=None, use_handcrafted=False, target_size=384):
        self.root_dir = root_dir
        self.transform = transform
        self.use_handcrafted = use_handcrafted
        self.target_size = target_size

        self.images = []
        self.labels = []

        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        authentic_seen = set()

        if os.path.isdir(root_dir):
            for img_name in os.listdir(root_dir):
                img_path = os.path.join(root_dir, img_name)

                if not os.path.isfile(img_path):
                    continue

                if os.path.splitext(img_name)[1].lower() not in self.valid_extensions:
                    continue

                parts = img_name.split('_')
                if len(parts) < 3:
                    continue

                base_id = parts[0]

                if '_F_' in img_name:
                    self.images.append(img_path)
                    self.labels.append(1)

                elif '_O_' in img_name:
                    if base_id not in authentic_seen:
                        self.images.append(img_path)
                        self.labels.append(0)
                        authentic_seen.add(base_id)

        print(f"[DATASET] {len(self.images)} images loaded")
        print(f"   Authentic: {sum(1 for l in self.labels if l == 0)} | Forged: {sum(1 for l in self.labels if l == 1)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.Resize((self.target_size, self.target_size))(image)
                image = transforms.ToTensor()(image)

            if self.use_handcrafted:
                pil_img = Image.open(img_path).convert('RGB')
                pil_img = transforms.Resize((224, 224))(pil_img)
                img_array = np.array(pil_img) / 255.0

                # initialize safely
                dct_feat = np.zeros(16)
                zernike_feat = np.zeros(7)
                lbp_feat = np.zeros(16)

                # extract individually
                try:
                    dct_feat = extract_dct_features(img_array)
                except Exception as e:
                    print(f"DCT error: {e}")

                try:
                    zernike_feat = extract_zernike_features(img_array)
                except Exception as e:
                    print(f"Zernike error: {e}")

                try:
                    lbp_feat = extract_lbp_features(img_array)
                except Exception as e:
                    print(f"LBP error: {e}")

                # enforce correct sizes
                dct_feat = np.pad(dct_feat, (0, max(0, 16 - len(dct_feat))))[:16]
                zernike_feat = np.pad(zernike_feat, (0, max(0, 7 - len(zernike_feat))))[:7]
                lbp_feat = np.pad(lbp_feat, (0, max(0, 16 - len(lbp_feat))))[:16]

                handcrafted = np.concatenate([dct_feat, zernike_feat, lbp_feat])

                handcrafted = torch.from_numpy(handcrafted).float()
                return image, handcrafted, label

            return image, label

        except Exception as e:
            print(f"Image load error: {img_path} - {e}")

            blank = torch.zeros(3, self.target_size, self.target_size)

            if self.use_handcrafted:
                return blank, torch.zeros(39), label

            return blank, label
