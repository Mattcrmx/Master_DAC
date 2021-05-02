import datetime
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from utils import read_temps, RNN, TempDatasetClassif, LightningRNN


class LightningRNNClassif(LightningRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        self.model = RNN(input_dim=1, latent=self.hparams.latent_dim, output_dim=self.hparams.n_cities,
                         encode=self.sigma_encode, decode=self.sigma_decode,
                         batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        N, T, d = batch_x.shape
        batch_x = batch_x.view(T, N, d)

        h = self.model(batch_x, device=self.device)
        y_hat = self.model.decode(h[-1])
        loss = self.criterion(y_hat, batch_y)
        train_acc = self.train_accuracy(y_hat, batch_y)

        self.log('train_acc', train_acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        N, T, d = batch_x.shape
        batch_x = batch_x.view(T, N, d)

        h = self.model(batch_x, device=self.device)
        y_hat = self.model.decode(h[-1])
        loss = self.criterion(y_hat, batch_y)
        val_acc = self.val_accuracy(y_hat, batch_y)

        self.log('val_acc', val_acc)
        self.log('val_loss', loss)
        return loss


def collate_fn(samples):
    seq, labels = zip(*samples)
    max_length, _ = seq[0].shape
    t_prime = random.randint(1, max_length)
    batch = []
    for sequence in seq:
        batch.append(sequence[:t_prime])
    return torch.stack(batch, dim=0), torch.stack(labels, dim=0)


if __name__ == '__main__':
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config/exo2.yaml'))
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(f'../runs/latent_{conf.latent_dim}_lr_{conf.lr}_batch_{conf.batch_size}_{time_tag}')

    # data loading
    data_train = read_temps(PROJECT_DIR / 'data/tempAMAL_train.csv')
    data_test = read_temps(PROJECT_DIR / 'data/tempAMAL_test.csv')
    # data standardization: fit on train dataset, and transform both train and test datasets
    mean, std = data_train.mean(), data_train.std()
    data_train = (data_train - mean) / std
    data_test = (data_test - mean) / std

    train_dataset = TempDatasetClassif(data_train[:, :conf.n_cities], max_length=conf.seq_length)
    val_dataset = TempDatasetClassif(data_test[:, :conf.n_cities], max_length=conf.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, drop_last=True, num_workers=4,
                                collate_fn=collate_fn)

    # create pl model
    rnn_pl = LightningRNNClassif(hparams=conf)

    # train model
    # trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger)
    trainer.fit(rnn_pl, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
