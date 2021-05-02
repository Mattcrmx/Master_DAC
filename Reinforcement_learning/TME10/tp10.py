import datetime
from math import sqrt

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler, LatentDimInterpolator
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import make_grid


class MLP(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_out),
        )

    def forward(self, x):
        return self.model(x)


class EncoderCNN(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # reshape into (N, C, H, W)
        x = self.cnn(x)  # now shape (N, 32, 4, 4)
        x = x.view(-1, 32*4*4)
        x = self.fc(x)
        return x


class DecoderCNN(nn.Module):
    def __init__(self, dim_z):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_z, 300),
            nn.ReLU(),
            nn.Linear(300, 4*4*32),
            nn.ReLU()
        )
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 4, 4)
        x = self.cnn(x)
        x = x.view(-1, 28 * 28)  # reshape into (N, dim_x)
        return x


class VAE(pl.LightningModule):
    def __init__(self, h_dim, z_dim, lr, decoder_dist, model_type):
        super().__init__()
        # used by pl callback TensorboardGenerativeModelImageSampler
        self.img_dim = (1, 28, 28)
        self.hparams.latent_dim = z_dim

        self.lr = lr
        self.x_dim = 28 * 28
        self.z_dim = z_dim
        self.decoder_dist = decoder_dist

        if model_type == 'mlp':
            # flattened input image x -> [mu(x), sigma_squared(x)],
            #   parameters of q(z|x), normal distribution
            self.encoder_mu = MLP(self.x_dim, h_dim, z_dim)
            self.encoder_var = MLP(self.x_dim, h_dim, z_dim)

            # latent vector z -> f(z), logits for the parameters of p_theta(x|z)
            # If p_theta is the normal distribution :
            #   mu = sigmoid(f(z)), var = 1
            # If p_theta is the bernoulli distribution :
            #   each pixel has parameter p = sigmoid(f(x)) in [0, 1]
            self.decoder = MLP(z_dim, h_dim, self.x_dim)
        elif model_type == 'cnn':
            self.encoder_cnn = EncoderCNN(dim_out=300)
            self.encoder_mu = nn.Sequential(
                self.encoder_cnn,
                nn.Linear(300, z_dim)
            )
            self.encoder_var = nn.Sequential(
                self.encoder_cnn,
                nn.Linear(300, z_dim)
            )
            self.decoder = DecoderCNN(z_dim)
        else:
            raise ValueError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_loss(self, batch, tag):
        x, _ = batch
        x = x.view(-1, self.x_dim)

        # our approximated distribution of the latent space q_phi(z|x)
        mu, var = self.encoder_mu(x), self.encoder_var(x)
        q_z = Normal(mu, var)

        # sample a batch of re-parameterized z latent variables
        z = q_z.rsample()
        x_logits = self.decoder(z)

        # chosen latent space distribution p(z) (standard normal)
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(var))

        # Regularization loss, to make q_phi(z|x) match p(z)
        # kl_div(q_phi(z|x) || p(z))
        kl_div = torch.distributions.kl_divergence(q_z, p_z)  # not reduced (N, z_dim)
        kl_div = kl_div.sum(dim=1).mean()  # average KL div for the batch

        # Reconstruction loss
        # -E[log(p_theta(x|z)]
        if self.decoder_dist == 'normal':
            # assuming p_theta(x|z) = Normal(sigmoid(x_logits), 1), this is equivalent to the batch mean of log(p_theta)
            recon_loss = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').sum(dim=1).mean()
        elif self.decoder_dist == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').sum(dim=1).mean()
        else:
            raise ValueError(f'decoder distribution {self.decoder_dist} not valid')

        if self.global_step % 100 == 0 or tag == 'val':
            self.log_images(x, x_logits, z, tag)

        loss = recon_loss + kl_div
        return loss, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self.get_loss(batch, tag='train')
        self.log('train_recon_loss', recon_loss.item())
        self.log('train_kl_div', kl_loss.item())
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss = self.get_loss(batch, tag='val')
        self.log('val_recon_loss', recon_loss.item())
        self.log('val_kl_div', kl_loss.item())
        self.log('val_loss', loss.item())
        return loss

    def log_images(self, x, x_logits, z, tag):
        x_hat = torch.sigmoid(x_logits)
        z_img = make_grid(z[:32].reshape(-1, 1, 1, self.z_dim).repeat(1, 3, 1, 1))
        self.logger.experiment.add_image(f'{tag}_latent_vectors', z_img, self.global_step)
        images_original = make_grid(x[:32].reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1))
        images_predict = make_grid(x_hat[:32].reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1))
        self.logger.experiment.add_images(f'{tag}_images', torch.stack((images_original, images_predict)),
                                          self.global_step)

    def forward(self, z):
        # (Used by pl callback LatentDimInterpolator to plot images)
        z = z.view(-1, self.z_dim)
        x_hat = torch.sigmoid(self.decoder(z))
        return x_hat.view(-1, 28, 28)


def main():
    conf = OmegaConf.load('config.yaml')
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    logger = TensorBoardLogger(f'runs/tp10-{conf.decoder_dist}-{conf.model}-z_dim-{conf.z_dim}-{time_tag}')
    print(f'Running with conf: \n{OmegaConf.to_yaml(conf)}')

    # inspiration :
    # https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

    # todo : read/add to the report
    # https://papers.nips.cc/paper/2019/file/f82798ec8909d23e55679ee26bb26437-Paper.pdf

    # Load MNIST data as [0,1] float tensors
    train_dataset = MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    val_dataset = MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size)

    model = VAE(h_dim=conf.h_dim, z_dim=conf.z_dim, lr=conf.lr, decoder_dist=conf.decoder_dist, model_type=conf.model)

    # train model
    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger,
                         callbacks=[LatentDimInterpolator(interpolate_epoch_interval=100)])
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    # Log hyper parameters to tensorboard
    conf_clean = {k: str(v) for (k, v) in conf.items()}
    logger.experiment.add_hparams(conf_clean, metric_dict={'score': 0.})


if __name__ == '__main__':
    main()
