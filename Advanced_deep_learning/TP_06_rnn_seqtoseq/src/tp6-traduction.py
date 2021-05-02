import datetime
import logging
import random
import re
import string
import time
import unicodedata
from pathlib import Path
from typing import List, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sentencepiece as spm
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

logging.basicConfig(level=logging.INFO)

PROJECT_DIR = Path(__file__).resolve().parents[1]
s_eng = spm.SentencePieceProcessor(model_file=str(PROJECT_DIR / 'model/en_segmentator_not_normalized.model'))
s_fra = spm.SentencePieceProcessor(model_file=str(PROJECT_DIR / 'model/fr_segmentator_not_normalized.model'))
# s_eng = spm.SentencePieceProcessor(model_file=str(PROJECT_DIR / 'model/english_segmentator.model'))
# s_fra = spm.SentencePieceProcessor(model_file=str(PROJECT_DIR / 'model/french_segmentator.model'))
PAD_ID = s_eng.pad_id()
UNK_ID = s_eng.unk_id()
SOS_ID = s_eng.bos_id()
EOS_ID = s_eng.eos_id()


def normalize(s):
    return re.sub(' +', ' ', "".join(c if c in string.ascii_letters else " "
                                     for c in unicodedata.normalize('NFD', s.lower().strip())
                                     if c in string.ascii_letters + " " + string.punctuation)).strip()


def masked_cross_entropy(logits, target):
    return nn.functional.cross_entropy(
        logits.permute(1, 2, 0),  # (N, num_classes, L)
        target.t(),  # (N, L)
        ignore_index=PAD_ID
    )


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": PAD_ID, "EOS": EOS_ID, "SOS": SOS_ID}
        if oov:
            self.word2id["__OOV__"] = UNK_ID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, UNK_ID)
        return self.word2id[word]

    def get_id(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return UNK_ID
            raise

    def __len__(self):
        return len(self.id2word)

    def get_word(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def get_words(self, idx: List[int], get_padded=True):
        if get_padded:
            return [self.get_word(i) for i in idx]
        else:
            return [self.get_word(i) for i in idx if i != PAD_ID]

    def get_words_as_str(self, idx: List[int]):
        return ' '.join(self.get_words(idx, get_padded=False))


class TraductorLightning(pl.LightningModule):
    def __init__(self, embedding_dim, voc_in: Union[Vocabulary, spm.SentencePieceProcessor],
                 voc_out: Union[Vocabulary, spm.SentencePieceProcessor], nb_layers, hidden_dim, lr,
                 teacher_forcing_tau,
                 log_freq_text,
                 bidirectional):
        super(TraductorLightning, self).__init__()
        self.lr = lr
        self.teacher_forcing_tau = teacher_forcing_tau
        self.log_freq_text = log_freq_text
        self.voc_in = voc_in
        self.voc_out = voc_out
        self.use_segmentation = isinstance(voc_in, spm.SentencePieceProcessor)
        self.gru_encoder = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=nb_layers,
                                  bidirectional=bidirectional)
        self.gru_decoder = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=nb_layers,
                                  bidirectional=bidirectional)
        self.embedding_encoder = nn.Embedding(num_embeddings=len(voc_in), embedding_dim=embedding_dim,
                                              padding_idx=PAD_ID)
        self.embedding_decoder = nn.Embedding(num_embeddings=len(voc_out), embedding_dim=embedding_dim,
                                              padding_idx=PAD_ID)
        self.linear_decoder = nn.Linear(hidden_dim, len(voc_out))

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def forward(self, x, y, teacher_forcing_ratio):
        max_len, N = y.shape
        x = self.embedding_encoder(x)
        _, h = self.gru_encoder(x)

        decoder_input = torch.tensor([SOS_ID] * N).view(1, N)  # (1, N)

        logits = []
        for t in range(max_len):
            # get next character logits
            decoder_input = self.embedding_decoder(decoder_input)  # (1, N, dim_emb)
            decoder_input = nn.functional.relu(decoder_input)
            _, h = self.gru_decoder(decoder_input, h)  # (nb_layers, N, h_dim)
            logits_t = self.linear_decoder(h[-1])  # (N, vocab_size)
            logits.append(logits_t)

            # use either the predicted character or the ground truth as the next decoder input
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = y[t].view(1, N)
            else:
                decoder_input = logits_t.argmax(dim=1).view(1, N)

        logits = torch.stack(logits, dim=0)  # (max_len, N, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        teacher_forcing_ratio = np.exp(-self.current_epoch / self.teacher_forcing_tau)
        logits = self(x, y, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = masked_cross_entropy(logits, y)

        if self.global_step % self.log_freq_text == 0:
            self.log_text(x, y, logits, tag='train')
        self.log('teacher_forcing_ratio', teacher_forcing_ratio)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, teacher_forcing_ratio=0)
        loss = masked_cross_entropy(logits, y)

        self.log_text(x, y, logits, tag='val')
        self.log('val_loss', loss)
        return loss

    def log_text(self, x, y, logits, tag):
        y_hat = logits.argmax(dim=2)
        if self.use_segmentation:
            input = self.voc_in.decode(x[:, 0].tolist())
            label = self.voc_out.decode(y[:, 0].tolist())
            predict = self.voc_out.decode(y_hat[:, 0].tolist())
        else:
            input = self.voc_in.get_words_as_str(x[:, 0])
            label = self.voc_out.get_words_as_str(y[:, 0])
            predict = self.voc_out.get_words_as_str(y_hat[:, 0])
        self.logger.experiment.add_text(f'{tag}_input', input, self.global_step)
        self.logger.experiment.add_text(f'{tag}_label', label, self.global_step)
        self.logger.experiment.add_text(f'{tag}_predict', predict, self.global_step)

    def on_train_epoch_end(self, outputs):
        n_in = len(self.voc_in)
        n_out = len(self.voc_out)
        labels_in = self.voc_in.id_to_piece(list(range(n_in)))
        labels_out = self.voc_out.id_to_piece(list(range(n_out)))
        self.logger.experiment.add_embedding(self.embedding_encoder.weight, metadata=labels_in,
                                             global_step=self.global_step, tag='emb_encoder')
        self.logger.experiment.add_embedding(self.embedding_decoder.weight, metadata=labels_out,
                                             global_step=self.global_step, tag='emb_decoder')


class SegmentationDataset(Dataset):
    def __init__(self, data, english_segmentator, french_segmentator, max_len=10, use_normalisation=True):
        self.data = data
        self.english_segmentator = english_segmentator
        self.french_segmentator = french_segmentator
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1:
                continue
            if use_normalisation:
                orig, dest = map(normalize, s.split("\t")[:2])
            else:
                orig, dest = s.split("\t")[:2]
            if len(orig) > max_len:
                continue
            self.sentences.append((
                torch.tensor(self.english_segmentator.encode(orig, out_type=int) + [self.english_segmentator.eos_id()]),
                torch.tensor(self.french_segmentator.encode(dest, out_type=int) + [self.french_segmentator.eos_id()])
            ))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


class TradDataset(Dataset):
    def __init__(self, data, voc_origin, voc_destination, max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s) < 1:
                continue
            orig, dest = map(normalize, s.split("\t")[:2])
            if len(orig) > max_len:
                continue
            self.sentences.append((torch.tensor([voc_origin.get_id(o) for o in orig.split(" ")] + [EOS_ID]),
                                   torch.tensor([voc_destination.get_id(o) for o in dest.split(" ")] + [EOS_ID])))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


def collate(batch):
    origin, destination = zip(*batch)
    return pad_sequence(origin, padding_value=PAD_ID), pad_sequence(destination, padding_value=PAD_ID)


def main():
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config/traduction.yaml'))
    time_tag = datetime.datetime.now().strftime(f'trad-layers-{conf.nb_layers}-emb_dim-{conf.embedding_dim}'
                                                f'-bidir-{conf.bidirectional}-%Y%m%d-%H%M%S')
    logger = TensorBoardLogger(f'../runs/{time_tag}')

    # --- data loading
    logging.info("Loading datasets...")

    with open(PROJECT_DIR / 'data/en-fra.txt') as f:
        lines = f.readlines()

    lines = [lines[x] for x in torch.randperm(len(lines))]
    # use only half of the dataset
    lines = lines[:int(0.3 * len(lines))]
    idxTrain = int(0.8 * len(lines))

    if conf.use_segmentation:
        voc_eng = s_eng
        voc_fra = s_fra
        dataset_train = SegmentationDataset("".join(lines[:idxTrain]), voc_eng, voc_fra, max_len=conf.max_len)
        dataset_val = SegmentationDataset("".join(lines[idxTrain:]), voc_eng, voc_fra, max_len=conf.max_len)
    else:
        voc_eng = Vocabulary(True)
        voc_fra = Vocabulary(True)
        dataset_train = TradDataset("".join(lines[:idxTrain]), voc_eng, voc_fra, max_len=conf.max_len)
        dataset_val = TradDataset("".join(lines[idxTrain:]), voc_eng, voc_fra, max_len=conf.max_len)
        logging.info("English Vocabulary size: %d", len(voc_eng))
        logging.info("French Vocabulary size: %d", len(voc_fra))

    train_dataloader = DataLoader(dataset_train, collate_fn=collate, batch_size=conf.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, collate_fn=collate, batch_size=conf.batch_size)

    trad_model = TraductorLightning(embedding_dim=conf.embedding_dim, voc_in=voc_eng, voc_out=voc_fra,
                                    nb_layers=conf.nb_layers, hidden_dim=conf.hidden_dim, lr=conf.lr,
                                    teacher_forcing_tau=conf.teacher_forcing_tau, log_freq_text=conf.log_freq_text,
                                    bidirectional=conf.bidirectional)

    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger)
    trainer.fit(trad_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
