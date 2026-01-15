from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class CycleGANDataset(Dataset):
    def __init__(self, root_image, transforms=None):
        self.root_image = root_image
        self.transforms = transforms
        self.dir_A = os.path.join(root_image, "A")
        self.dir_B = os.path.join(root_image, "B")
        
        # Check if directories exist
        if not os.path.exists(self.dir_A) or not os.path.exists(self.dir_B):
            raise FileNotFoundError(f"Directories 'A' and/or 'B' not found in {root_image}")

        self.image_A = os.listdir(self.dir_A)
        self.image_B = os.listdir(self.dir_B)
        self.length_dataset = max(len(self.image_A), len(self.image_B))
        self.image_A_len = len(self.image_A)
        self.image_B_len = len(self.image_B)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_A_name = self.image_A[index % self.image_A_len]
        image_B_name = self.image_B[index % self.image_B_len]

        path_A = os.path.join(self.dir_A, image_A_name)
        path_B = os.path.join(self.dir_B, image_B_name)

        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")

        if self.transforms:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)

        return img_A, img_B
