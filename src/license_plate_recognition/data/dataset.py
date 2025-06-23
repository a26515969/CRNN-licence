import os
from pathlib import Path
from typing import Union

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose, ToTensor

import license_plate_recognition.data.utils as utils
from torch.nn.utils.rnn import pad_sequence

class LicencePlateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        img_dir: Path,
        dictionary: utils.Dictionary,
        label_file: Union[str, Path] = None,
        img_size: tuple[int, int] = (128, 32),
        train: bool = True,
        transform: Union[None, Compose, ToTensor] = None,
    ):
        super().__init__()

        self.img_dir = img_dir
        self.dictionary = dictionary
        self.img_size = img_size
        self.transform = transform

        # 讀入 label 對照表
        if label_file is None:
            raise ValueError("label_file must be provided.")

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.label_map = {}
        for line in lines:
            if line.strip():
                fname, label = line.strip().split("\t")
                self.label_map[fname] = label

        self.images = [os.path.join(self.img_dir, fname) for fname in self.label_map]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]
        fname = os.path.basename(image_path)

        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.transform:
            img = self.transform(img)

        label_str = self.label_map[fname]
        label_encoded = torch.LongTensor([self.dictionary.char2idx[char] for char in label_str])
        return img, label_encoded
    
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return images, targets