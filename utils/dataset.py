import torch
from torch.utils.data.dataset import Dataset
import glob
import os
import cv2

class ImageNetDataset(Dataset):

    def __init__(self, root, train=True, transform=None, cls2idx=None, exts=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.exts = exts if exts else ['.jpeg', '.jpg', '.png'] 

        labels = sorted(os.listdir(root))
        if not labels:
            raise ValueError(f"No class folders found in {root}")

        self.cls2idx = {label: idx for idx, label in enumerate(labels)} if train else cls2idx

        self.img_path_lst = self._load_image_paths()

    def _load_image_paths(self):
        img_paths = []
        for ext in self.exts:
            img_paths.extend(glob.glob(os.path.join(self.root, '**', f'*{ext}'), recursive=True))
        return img_paths

    def __getitem__(self, index):

        img_path = self.img_path_lst[index]
        img = cv2.imread(img_path)  

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.cls2idx.get(label_name, -1)

        if label == -1:
            raise ValueError(f"Label '{label_name}' not found in cls2idx mapping.")

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.img_path_lst)