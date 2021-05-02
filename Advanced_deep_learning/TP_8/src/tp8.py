import datetime
import logging
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from torch.distributions import Categorical
from datamaestro import prepare_dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


class MNIST(Dataset):
    def __init__(self, images, labels, device, transform=None):
        # copy the np arrays to remove the UserWarning, as they are not writeable
        self.images = torch.tensor(images.copy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.copy(), dtype=torch.int64)
        self.images /= 255.0
        self.transform = transform

        _, self.width, self.height = self.images.shape
        self.n_features = self.width * self.height

        # Use GPU if available
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)

    def __getitem__(self, index):
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.images)


class Net(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr, log_freq_grad, regularization, weight_decay=None,
                 l1_reg=None, dropout=False, batchnorm=False, layernorm=False):
        super(Net, self).__init__()
        self.n_features = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.log_freq_grad = log_freq_grad
        self.lr = lr
        self.regularization = regularization
        self.weight_decay = weight_decay
        self.l1_reg = l1_reg
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if dropout:
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.5)
        elif batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(hidden_dim)
        elif layernorm:
            self.layernorm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.batchnorm:
            l1 = self.linear1(x)
            b1 = self.batchnorm_layer(l1)
            l2 = self.linear2(b1)
            return self.linear3(l2)
        elif self.dropout:
            l1 = self.linear1(x)
            d1 = self.dropout1(l1)
            l2 = self.linear2(d1)
            d2 = self.dropout2(l2)
            return self.linear3(d2)
        elif self.layernorm:
            l1 = self.linear1(x)
            ln1 = self.layernorm_layer(l1)
            l2 = self.linear2(ln1)
            return self.linear3(l2)
        else:
            l1 = self.linear1(x)
            l2 = self.linear2(l1)
            return self.linear3(l2)

    def configure_optimizers(self):
        if self.regularization == 'L2':
            return torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        by default, lightning executes this under the scope model.train(),
        no need to repeat that line.
        """
        x, y = batch
        x = x.reshape(-1, self.n_features)
        logits = self(x)

        if self.regularization == 'L1':
            l1_penalty = 0
            for param in self.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            loss = nn.functional.cross_entropy(logits, y) + self.l1_reg * l1_penalty
        else:
            loss = nn.functional.cross_entropy(logits, y)

        acc = self.train_acc(logits, y)

        self.log('train_acc', acc)
        self.log('train_loss', loss)

        # log the entropy of the output comparatively to a random model
        if self.global_step % self.log_freq_grad == 0 and self.log_freq_grad > 0:
            random_logits = torch.randn_like(logits)
            self.logger.experiment.add_histogram('entropy_output', Categorical(logits=logits).entropy(),
                                                 self.global_step)
            self.logger.experiment.add_histogram('entropy_random_output', Categorical(logits=random_logits).entropy(),
                                                 self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        same as above, the model is already in 'eval mode' : model.eval() and therefore there is no issue with the
        regularization layers like dropout.
        """
        x, y = batch
        x = x.reshape(-1, self.n_features)
        logits = self(x)

        loss = nn.functional.cross_entropy(logits.data, y.data)
        acc = self.train_acc(logits.data, y.data)

        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return loss

    def on_after_backward(self):
        global_step = self.global_step
        if global_step % self.log_freq_grad == 0 and self.log_freq_grad > 0:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('src/config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'mnist-hidden-dim{conf.hidden_dim}-reg{conf.regularization}'
                                                f'-%Y%m%d-%H%M%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = TensorBoardLogger(f'../runs/{time_tag}')

    # transformations :
    trsfm = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=45),
         transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor()])

    # data loading
    logging.info("Loading datasets...")
    ds = prepare_dataset("com.lecun.mnist")

    train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
    train_dataset = MNIST(train_images, train_labels, device=device)
    new_len = conf.split_ratio * len(train_dataset)
    train_dataset, val_dataset = random_split(dataset=train_dataset,
                                              lengths=[int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))],
                                              generator=torch.Generator().manual_seed(42069))
    train_dataset, val_dataset = random_split(dataset=train_dataset,
                                              lengths=[int(new_len), int(len(train_dataset) - new_len)],
                                              generator=torch.Generator().manual_seed(42069))
    test_dataset = MNIST(test_images, test_labels, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size)
    dataloader_test = DataLoader(test_dataset, batch_size=conf.batch_size)

    model = Net(input_dim=test_dataset.n_features, hidden_dim=conf.hidden_dim, output_dim=10, lr=conf.lr,
                log_freq_grad=conf.log_freq_grad, regularization=conf.regularization, weight_decay=conf.weight_decay,
                l1_reg=conf.l1_reg, dropout=conf.dropout, batchnorm=conf.batchnorm, layernorm=conf.layernorm)

    # train model
    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
