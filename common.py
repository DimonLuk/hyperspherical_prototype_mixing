import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def calculate_mean_std(ds, shape, data_extractor):
    mean = torch.zeros(shape)
    std = torch.zeros(shape)

    for data in ds:
        img, _ = data_extractor(data)
        mean += img

    mean = mean.sum(dim=[1, 2]) / (len(ds) * shape[1] * shape[2])

    for data in ds:
        img, _ = data_extractor(data)
        std += (img - mean[:, None, None]) ** 2

    std = torch.sqrt(std.sum(dim=[1, 2]) / (len(ds) * shape[1] * shape[2]))

    return mean, std


def dump_dataset(ds, split, directory, data_extractor):
    result = []

    for index, data in enumerate(ds):
        img, label = data_extractor(data)

        path = f"./{directory}/data/{split}/{index}.pt"
        img = (img * 255).to(dtype=torch.uint8)

        torch.save(img, path)

        result.append((path, label))

    with open(f"{directory}/{split}_ds.pkl", "wb") as file:
        pickle.dump(result, file)


def create_centroids(num_classes, dim, lr, steps, directory):
    centroids = torch.empty((num_classes, dim))
    nn.init.xavier_uniform_(centroids)
    centroids.requires_grad = True

    charges = torch.ones((num_classes, 1))

    optimizer = torch.optim.SGD([centroids], lr=lr)

    for _ in (pbar := tqdm(range(int(steps)))):
        optimizer.zero_grad()

        normed_centroids = F.normalize(centroids, p=2, dim=1)
        cosines = normed_centroids @ normed_centroids.T

        sims = (cosines + 1) / 2
        sims.fill_diagonal_(0)

        loss = (charges @ charges.T) / (1 - sims)
        loss.fill_diagonal_(0)
        loss = loss.sum()

        pbar.set_postfix({"loss": f"{float(loss):.5f}"})

        loss.backward()
        optimizer.step()

    centroids = centroids.detach()
    centroids.requires_grad = False
    centroids.grad = None
    centroids = F.normalize(centroids, p=2, dim=1)
    centroids = centroids.T
    torch.save(centroids, f"{directory}/centroids.pt")
