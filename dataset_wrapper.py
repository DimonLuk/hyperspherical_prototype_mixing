import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, data, mean, std, augment):
        super().__init__()

        self.data = data

        self.normalize = v2.Normalize(mean=mean, std=std)

        self.augment = augment

        self.augmentations = v2.RandAugment(3, 9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]

        img = torch.load(path)
        img = img.to(dtype=torch.float32) / 255

        if self.augment:
            img = self.augmentations(img)

        img = self.normalize(img)

        return img, label
