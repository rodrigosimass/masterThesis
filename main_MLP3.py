import os
import torch
import pytorch_lightning as pl
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


class MLP3(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(21 * 21 * 20, 4000),
            nn.ReLU(),
            nn.Linear(4000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 28 * 28),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())

    pl.seed_everything(42)
    mlp = MLP3()
    trainer = pl.Trainer(
        auto_scale_batch_size="power", gpus=0, deterministic=True, max_epochs=5
    )
    trainer.fit(mlp, DataLoader(dataset))
