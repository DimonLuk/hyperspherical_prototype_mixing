import random

import torch
import torchvision
import torchvision.transforms.v2 as v2

from common import calculate_mean_std, dump_dataset, create_centroids

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    train_ds = torchvision.datasets.CIFAR100(
        "./cifar100/data/cifar100",
        train=True,
        transform=v2.Compose([v2.Resize((64, 64)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        download=True,
    )

    mean, std = calculate_mean_std(train_ds, (3, 64, 64))
    print(mean)
    print(std)

    dump_dataset(train_ds, "train", "cifar100")

    val_ds = torchvision.datasets.CIFAR100(
        "./cifar100/data/cifar100",
        train=False,
        transform=v2.Compose([v2.Resize((64, 64)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        download=True,
    )

    dump_dataset(val_ds, "val", "cifar100")

    create_centroids(100, 512, 1e2, 3e5, "cifar100")
