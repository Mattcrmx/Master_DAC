from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import read_temps, RNN, TempDatasetForecast, LightningRNN
from pathlib import Path
import datetime
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class LightningRNNForecast(LightningRNN):
    def __init__(self, predict_window, **kwargs):
        super().__init__(**kwargs)
        self.predict_window = predict_window

        # in this case, this should be the same as the loss if we are using MSE loss, but do it as a sanity check
        self.train_mse = pl.metrics.MeanSquaredError()
        self.val_mse = pl.metrics.MeanSquaredError()

        self.model = RNN(input_dim=1, latent=self.hparams.latent_dim, output_dim=1,
                         encode=self.sigma_encode, decode=self.sigma_decode,
                         batch_size=self.hparams.batch_size)

    def step(self, batch, batch_idx):
        batch_x, batch_y, batch_t, batch_city = batch
        batch_x = batch_x.transpose(0, 1)
        batch_y = batch_y.transpose(0, 1)

        h = self.model(batch_x, device=self.device)
        h_last = h[-1]
        batch_y_hat = []
        for t in range(self.predict_window):
            y_hat = self.model.decode(h_last)
            h_last = self.model.one_step(y_hat, h_last)
            batch_y_hat.append(y_hat)
        batch_y_hat = torch.stack(batch_y_hat)

        return batch_x, batch_y_hat, batch_y, batch_t, batch_city

    def training_step(self, batch, batch_idx):
        batch_x, batch_y_hat, batch_y, batch_t, batch_city = self.step(batch, batch_idx)
        loss = self.criterion(batch_y_hat, batch_y)
        train_acc = self.train_mse(batch_y_hat,
                                   batch_y)  # todo: transpose tensors to (N, L, 1) even though in theory this works

        if self.global_step % self.hparams.log_freq_fig == 0:
            fig = self.plot_figures(batch_x, batch_y_hat, batch_y, batch_t, batch_city)
            self.logger.experiment.add_figure('figures', fig, global_step=self.global_step)

        self.log('train_mse', train_acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, batch_y_hat, batch_y, _, _ = self.step(batch, batch_idx)
        loss = self.criterion(batch_y_hat, batch_y)
        val_acc = self.val_mse(batch_y_hat, batch_y)

        self.log('val_mse', val_acc)
        self.log('val_loss', loss)
        return loss

    @staticmethod
    def plot_figures(batch_x, batch_y_hat, batch_y, batch_t, batch_city):
        # plot the 8 first images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 3))
        for idx in range(8):
            ax = fig.add_subplot(2, 4, idx + 1)
            x = batch_x[:, idx, 0].detach().numpy()
            y = batch_y[:, idx, 0].detach().numpy()
            y_hat = batch_y_hat[:, idx, 0].detach().numpy()
            ax.plot(x, '.-', label='input')
            timesteps_forecast = list(range(len(x), len(x) + len(y)))
            ax.set_title(f't={batch_t[idx]}, city={batch_city[idx]}')
            ax.plot(timesteps_forecast, y, '.-', label='label')
            ax.plot(timesteps_forecast, y_hat, '.-', label='forecast')
        plt.legend()
        return fig


if __name__ == '__main__':

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    args = OmegaConf.load(PROJECT_DIR.joinpath('config/exo3.yaml'))

    # args = parser.parse_args()
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(f'../runs/latent_{args.latent_dim}_lr_{args.lr}_batch_{args.batch_size}_{time_tag}')

    # data loading
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    data_train = read_temps(PROJECT_DIR / 'data/tempAMAL_train.csv')
    data_test = read_temps(PROJECT_DIR / 'data/tempAMAL_test.csv')
    # data standardization: fit on train dataset, and transform both train and test datasets
    mean, std = data_train.mean(), data_train.std()
    data_train = (data_train - mean) / std
    data_test = (data_test - mean) / std

    train_dataset = TempDatasetForecast(data_train[:, :args.n_cities], seq_length=args.seq_length,
                                        predict_window=args.predict_window)
    test_dataset = TempDatasetForecast(data_test[:, :args.n_cities], seq_length=args.seq_length,
                                       predict_window=args.predict_window)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4)

    # create pl model
    rnn_pl = LightningRNNForecast(hparams=args, predict_window=args.predict_window)
    x_logger = torch.stack([train_dataset[i][0] for i in range(4)])  # log the graph of the model
    logger.experiment.add_graph(rnn_pl.model, x_logger)

    # train model
    trainer = pl.Trainer(**OmegaConf.to_container(args.trainer), logger=logger)
    trainer.fit(rnn_pl, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
