import datetime
import logging
from pathlib import Path
from typing import List

import torch
from datamaestro import prepare_dataset
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)

PAD_ID = 0
OOV_ID = 1


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
        self.word2id = {"PAD": PAD_ID}
        self.id2word = ["PAD"]
        if oov:
            self.word2id["__OOV__"] = OOV_ID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, OOV_ID)
        return self.word2id[word]

    def get_id(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                word_id = len(self.id2word)
                self.word2id[word] = word_id
                self.id2word.append(word)
                return word_id
            if self.oov:
                return OOV_ID
            raise

    def __len__(self):
        return len(self.id2word)

    def get_word(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def get_words(self, idx: List[int]):
        return [self.get_word(i) for i in idx]

    def get_words_as_str(self, idx: List[int]):
        return ' '.join(self.get_words(idx))


class TaggingDataset(Dataset):
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get_id(token["form"], adding) for token in s],
                                   [tags.get_id(token["upostag"], adding) for token in s]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]


class TaggingModel(pl.LightningModule):
    def __init__(self, word_voc: Vocabulary, tag_voc: Vocabulary, embedding_dim: int, num_layers: int, lr: float,
                 log_freq_text: int):
        super().__init__()
        self.word_voc = word_voc
        self.tag_voc = tag_voc
        self.lr = lr
        self.log_freq_text = log_freq_text
        self.embedding = nn.Embedding(num_embeddings=len(word_voc), embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=len(tag_voc), num_layers=num_layers)

        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        (x, lens), y,  = batch
        logits = self(x, lens)

        loss = nn.functional.cross_entropy(logits.data, y.data)
        acc = self.train_acc(logits.data, y.data)

        if self.global_step % self.log_freq_text == 0:
            self.log_text(x, y, logits, tag='train')
        self.log('train_acc', acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, lens), y,  = batch
        logits = self(x, lens)

        loss = nn.functional.cross_entropy(logits.data, y.data)
        acc = self.train_acc(logits.data, y.data)

        self.log_text(x, y, logits, tag='val')
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return loss

    def forward(self, x, lens):
        """

        Args:
            x: (L_max, N) padded input tensor of indices
            lens: initial lengths of the N input sequences, before the padding

        Returns:
            output: (sum(L_i)*N, tag_size) packed output tensor of logits
        """
        # one-hot padded input to embedded vector space
        x = self.embedding(x)
        # pack input before feeding the RNN
        x = pack_padded_sequence(x, lens, enforce_sorted=False)

        output, _ = self.lstm(x)
        return output

    def log_text(self, x, y, logits, tag):
        y, lens = pad_packed_sequence(y)
        logits, _ = pad_packed_sequence(logits)
        first_seq_len = lens[0]
        y_hat = logits.argmax(dim=2)
        words = self.word_voc.get_words_as_str(x[:first_seq_len, 0])
        tags_label = self.tag_voc.get_words_as_str(y[:first_seq_len, 0])
        tags_predict = self.tag_voc.get_words_as_str(y_hat[:first_seq_len, 0])
        self.logger.experiment.add_text(f'{tag}_words', words, self.global_step)
        self.logger.experiment.add_text(f'{tag}_label', tags_label, self.global_step)
        self.logger.experiment.add_text(f'{tag}_predict', tags_predict, self.global_step)


def collate(samples: List[List[int]]):
    """
    Collate using pad_sequence and pack_sequence

    Args:
        samples:

    Returns:
        (x, lens), y - where
            x: (L, N) padded input tensor of word indices
            lens: initial lengths of the N input sequences, before the padding
            y: flat packed output tensor of tag labels
    """
    words = [torch.LongTensor(sample[0]) for sample in samples]
    tags = [torch.LongTensor(sample[1]) for sample in samples]
    # lens of initial sequences, before the padding (should be identical for words and tags)
    lens = [len(seq) for seq in words]

    words = pad_sequence(words, padding_value=PAD_ID)
    tags = pack_sequence(tags, enforce_sorted=False)
    return (words, lens), tags


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config/tagging.yaml'))
    time_tag = datetime.datetime.now().strftime(f'tag-layers-{conf.num_layers}-emb_dim-{conf.embedding_dim}'
                                                f'-%Y%m%d-%H%M%S')
    logger = TensorBoardLogger(f'../runs/{time_tag}')

    # --- data loading
    logging.info("Loading datasets...")
    words = Vocabulary(oov=True)
    tags = Vocabulary(oov=False)
    # Format de sortie décrit dans https://pypi.org/project/conllu/
    ds = prepare_dataset('org.universaldependencies.french.gsd')
    dataset_train = TaggingDataset(ds.train, words, tags, adding=True)
    dataset_val = TaggingDataset(ds.validation, words, tags, adding=True)
    dataset_test = TaggingDataset(ds.test, words, tags, adding=False)
    logging.info("Vocabulary size: %d", len(words))
    train_dataloader = DataLoader(dataset_train, collate_fn=collate, batch_size=conf.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, collate_fn=collate, batch_size=conf.batch_size)
    dataloader_test = DataLoader(dataset_test, collate_fn=collate, batch_size=conf.batch_size)
    # todo: test step

    # --- create and train model
    tag_model = TaggingModel(word_voc=words, tag_voc=tags, embedding_dim=conf.embedding_dim, num_layers=conf.num_layers,
                             lr=conf.lr, log_freq_text=conf.log_freq_text)

    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger)
    trainer.fit(tag_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
