import datetime
import gzip
import logging
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tp7_preprocess import TextDataset, Batch
import torch.nn.functional as F
import pytorch_lightning as pl

seed = 5
pl.seed_everything(seed)

logging.basicConfig(level=logging.INFO)


class Classifier(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        #  TODO: use (fine-tuned) pre-trained embeddings from word2vec
        # https://medium.com/@rohit_agrawal/using-fine-tuned-gensim-word2vec-embeddings-with-torchtext-and-pytorch-17eea2883cd
        # todo:
        #   - multiple kernels
        #   - batch_norm ? PReLU ?
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = nn.Conv1d(in_channels=emb_dim, out_channels=200, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=200, out_channels=100, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(100, 3)

    def forward(self, x):
        N, L = x.shape
        x = self.embedding(x)  # (N, L, emb_dim)
        x = torch.transpose(x, 1, 2)  # (N, emb_dim, L)
        x = self.conv1(x)  # (N, channels_out_1, L_1)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)  # (N, channels_out_2, L_2)
        x = torch.max(x, dim=2)  # (N, channels_out_2)  same than max-pooling over time ?
        x = self.fc(x)  # (N, out_classes=3)
        return x


class Model(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim, lr):
        super().__init__()
        self.lr = lr
        self.classifier = Classifier(vocab_size, emb_dim)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.lr)

    def training_step(self, batch: Batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)
        loss = F.cross_entropy()


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config/tp7.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    logger = TensorBoardLogger(f'../runs/{time_tag}')

    # Load datasets and tokenizer generated with the tp7_preprocess file
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(PROJECT_DIR.joinpath(f'models/wp{conf.vocab_size}.model')))

    def load_data(mode) -> TextDataset:
        with gzip.open(str(PROJECT_DIR.joinpath(f'models/{mode}-{conf.vocab_size}.pth')), "rb") as fp:
            return torch.load(fp)

    dataset_test = load_data('test')
    dataset_train = load_data('train')
    train_size = len(dataset_train) - conf.val_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, conf.val_size])
    logging.info("Datasets: train=%d, val=%d, test=%d", train_size, conf.val_size, len(dataset_test))
    logging.info("Vocabulary size: %d", conf.vocab_size)
    dataloader_train = DataLoader(dataset_train, batch_size=conf.batch_size, collate_fn=TextDataset.collate)
    dataloader_val = DataLoader(dataset_val, batch_size=conf.batch_size, collate_fn=TextDataset.collate)
    dataloader_test = DataLoader(dataset_test, batch_size=conf.batch_size, collate_fn=TextDataset.collate)

    # data source : http://help.sentiment140.com/for-students
    # models :
    # - https://arxiv.org/pdf/1408.5882.pdf


if __name__ == '__main__':
    main()
