import random

import torch
import torchvision.transforms.v2 as v2
from datasets import load_dataset

from common import calculate_mean_std, create_centroids, dump_dataset

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_ds = load_dataset("Maysee/tiny-imagenet", split="train")

    mean, std = calculate_mean_std(
        train_ds,
        (3, 64, 64),
        lambda x: (transforms(x["image"]), x["label"]),
    )
    print(mean)
    print(std)

    dump_dataset(
        train_ds,
        "train",
        "tiny_imagenet",
        lambda x: (transforms(x["image"]), x["label"]),
    )

    val_ds = load_dataset("Maysee/tiny-imagenet", split="valid")

    dump_dataset(
        val_ds,
        "val",
        "tiny_imagenet",
        lambda x: (transforms(x["image"]), x["label"]),
    )

    create_centroids(200, 512, 1e2, 3e5, "tiny_imagenet")
