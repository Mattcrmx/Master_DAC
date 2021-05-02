import datetime
import string
import unicodedata
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils import SpeechDataset, RNN, LightningRNN

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(1, len(LETTRES) + 1), LETTRES))
id2lettre[0] = ''  # NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))
ALPHABET_SIZE = len(id2lettre)


def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)


def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])


def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TextGenerationRNN(LightningRNN):
    def __init__(self, alphabet_size, code2string, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = self.hparams.seq_length
        self.alphabet_size = alphabet_size
        self.code2string = code2string
        # our model takes the previous one hot encoded character, and outputs the logits of the next character
        # note: we could also have a separate embedding layer, and feed the output of the embedding as an input to the
        # RNN (but given the small alphabet_size as we are working with characters and not words, we considered the
        # RNN could do all the work)
        self.model = RNN(input_dim=alphabet_size, latent=self.hparams.latent_dim, output_dim=alphabet_size,
                         encode=self.sigma_encode, decode=None, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        N, L = batch.shape
        h_t = torch.zeros(N, self.hparams.latent_dim)

        loss = 0
        predicted = [batch[:, 0]]  # starting character
        for t in range(L - 1):
            # using teacher forcing (input character is from the
            # text sequence, not from the last prediction of the model)
            x_t = batch[:, t]
            # note: this step would not be necessary if we were using torch.nn.Embedding
            x_t = torch.nn.functional.one_hot(x_t, num_classes=self.alphabet_size).float()

            h_t = self.model.one_step(x_t, h_t)
            y_logits = self.model.decode(h_t)
            y = batch[:, t + 1]
            loss += self.criterion(y_logits, y)

            # save and log characters (encoded as ints)
            predicted.append(y_logits.argmax(dim=1))

        if self.global_step % self.hparams.log_freq_text == 0:
            text_label = self.code2string(batch[0, :])
            # shape (L, N)
            text_predict = torch.stack(predicted)
            text_predict = self.code2string(text_predict[:, 0])[:len(text_label)]
            self.logger.experiment.add_text(f'train_predict', text_predict, self.global_step)
            self.logger.experiment.add_text(f'train_label', text_label, self.global_step)

        self.log('train_loss', loss)
        return loss


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config/exo4.yaml'))
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(f'../runs/exo4-latent-{conf.latent_dim}-{time_tag}')

    # --- data loading
    with open(PROJECT_DIR / 'data/trump_full_speech.txt') as f:
        data_str = f.read()
    data_str = ''.join([line for line in data_str])
    # Take only the first third of the dataset
    data_str = data_str[:len(data_str) // 3]
    # split into train and validation data
    size_train = int(0.7 * len(data_str))
    train_dataset = SpeechDataset(string2code(data_str[:size_train]), seq_length=conf.seq_length)
    val_dataset = SpeechDataset(string2code(data_str[size_train:]), seq_length=conf.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, drop_last=True, num_workers=4)
    # todo: add a validation step to actually use the validation dataset

    rnn_pl = TextGenerationRNN(alphabet_size=ALPHABET_SIZE, code2string=code2string, hparams=conf)

    # train model
    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger)
    trainer.fit(rnn_pl, train_dataloader=train_dataloader)  # , val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
