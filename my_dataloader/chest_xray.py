import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ChestXrayDataset(Dataset):
    """
    Dataset for Chest X-ray images structured as:
        chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
    
    Args:
        data_path (str): Root directory containing the 'chest_xray' folder.
        split (str): One of 'train', 'val', or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_path: str, split: str, transform: Optional[Callable] = None):
        assert split in ['train', 'val', 'test'], "split must be one of ['train', 'val', 'test']"
        self.root = Path(data_path) / "chest_xray" / split
        self.transform = transform
        self.samples = []

        for label_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = self.root / label_name
            label = 0 if label_name == 'NORMAL' else 1
            for img_file in class_dir.glob('*.jpeg'):
                self.samples.append((img_file, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
