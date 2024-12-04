import pickle
import random

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, SumMetric

from cifar10.custom_models import SmallResnet
from dataset_wrapper import DatasetWrapper
import torchvision.transforms.v2 as v2


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

        cutmix = v2.CutMix(num_classes=10)
        mixup = v2.MixUp(num_classes=10)
        self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def training_step(self, batch):
        x, labels = batch
        x, labels = self.cutmix_or_mixup(x, labels)

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

    with open("cifar10/train_ds.pkl", "rb") as file:
        train_ds = pickle.load(file)

    with open("cifar10/val_ds.pkl", "rb") as file:
        val_ds = pickle.load(file)

    train_ds = DatasetWrapper(
        train_ds,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
        augment=True,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    val_ds = DatasetWrapper(
        val_ds,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
        augment=False,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    model = SmallResnet()

    wrapper = LightningWrapper(model, torch.load("cifar10/centroids.pt"))

    trainer = L.Trainer(
        accelerator="cuda",
        devices=[0],
        logger=TensorBoardLogger(
            "cifar10/logs",
            "mixed_prototypes_softmax_with_linear",
        ),
        max_epochs=200,
    )
    trainer.fit(wrapper, train_dataloaders=train_dl, val_dataloaders=val_dl)
