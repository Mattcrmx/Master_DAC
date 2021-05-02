import optuna
from pytorch_lightning import Callback
from optuna.integration import PyTorchLightningPruningCallback
from tp8 import Net, MNIST
import datetime
import logging
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from torch.distributions import Categorical
from datamaestro import prepare_dataset
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
conf = OmegaConf.load(PROJECT_DIR.joinpath('src/config.yaml'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loading
logging.info("Loading datasets...")
ds = prepare_dataset("com.lecun.mnist")

train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
train_dataset = MNIST(train_images, train_labels, device=device)
new_len = conf.split_ratio * len(train_dataset)

train_dataset, val_dataset = random_split(dataset=train_dataset,
                                          lengths=[int(new_len), int(len(train_dataset) - new_len)],
                                          generator=torch.Generator().manual_seed(42069))
test_dataset = MNIST(test_images, test_labels, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size)


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(PROJECT_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
    )

    # study parameters
    # weight_decay = trial.suggest_float('weight_decay', low=0.0001, high=0.1)
    l1_reg = trial.suggest_float('l1_reg_coef', low=0.0001, high=0.1)

    model = Net(input_dim=test_dataset.n_features, hidden_dim=conf.hidden_dim, output_dim=10, lr=conf.lr,
                log_freq_grad=conf.log_freq_grad, regularization=conf.regularization, weight_decay=conf.weight_decay,
                l1_reg=l1_reg, dropout=conf.dropout, batchnorm=conf.batchnorm, layernorm=conf.layernorm)

    # train model
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(max_epochs=100, logger=False,
                         callbacks=[metrics_callback, PyTorchLightningPruningCallback(trial, monitor="val_loss")],
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    return metrics_callback.metrics[-1]["val_loss"].item()


if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
