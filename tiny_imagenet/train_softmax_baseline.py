import pickle
import random

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, SumMetric
from torchvision.models import resnet34

from dataset_wrapper import DatasetWrapper


class LightningWrapper(L.LightningModule):
    def __init__(self, model, centroids):
        super().__init__()

        self.model = model

        self.register_buffer(
            "centroids",
            centroids,
        )

        self.val_loss = SumMetric()
        self.val_accuracy = MeanMetric()

    def training_step(self, batch):
        x, labels = batch

        embeddings = F.normalize(self.model(x), p=2, dim=1)

        logits = embeddings @ self.centroids

        loss = F.cross_entropy(logits, labels)

        self.log(
            "train/loss",
            loss.detach(),
            prog_bar=True,
            sync_dist=True,
            on_step=True,
        )

        return loss

    def validation_step(self, batch):
        x, labels = batch
        bi = torch.arange(x.shape[0])

        embeddings = F.normalize(self.model(x), p=2, dim=1)

        logits = embeddings @ self.centroids
        cosines = logits[bi, labels]

        loss = (1 - cosines) ** 2

        accuracy = torch.argmax(logits, dim=1) == labels

        self.val_loss.update(loss)
        self.val_accuracy.update(accuracy)

    def on_validation_epoch_end(self):
        self.log(
            "val/loss",
            self.val_loss.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.val_loss.reset()

        self.log(
            "val/accuracy",
            self.val_accuracy.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


if __name__ == "__main__":
    BATCH_SIZE = 128
    random.seed(42)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    with open("tiny_imagenet/train_ds.pkl", "rb") as file:
        train_ds = pickle.load(file)

    with open("tiny_imagenet/val_ds.pkl", "rb") as file:
        val_ds = pickle.load(file)

    train_ds = DatasetWrapper(
        train_ds,
        mean=[0.4802, 0.4480, 0.3975],
        std=[0.2770, 0.2691, 0.2821],
        augment=True,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    val_ds = DatasetWrapper(
        val_ds,
        mean=[0.4802, 0.4480, 0.3975],
        std=[0.2770, 0.2691, 0.2821],
        augment=False,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    model = resnet34()
    model.fc = nn.Identity()

    wrapper = LightningWrapper(model, torch.load("tiny_imagenet/centroids.pt"))

    trainer = L.Trainer(
        accelerator="cuda",
        devices=[0],
        logger=TensorBoardLogger(
            "tiny_imagenet/logs",
            "baseline_softmax",
        ),
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(
                monitor="val/accuracy",
                mode="max",
                save_top_k=1,
            ),
        ],
    )
    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=val_dl)
