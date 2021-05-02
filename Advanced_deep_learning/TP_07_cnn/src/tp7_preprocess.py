import array
import csv
import gzip
import logging
import re
import shutil
from collections import namedtuple
from pathlib import Path

import click
import sentencepiece as spm
import torch
from datamaestro import prepare_dataset
from datamaestro.data.csv import Generic as CSVData
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

PROJECT_DIR = Path(__file__).parents[1]
RE_URL = re.compile(r"(?:\@|https?\://)\S+")
RE_MENTION = re.compile(r"(?:@)\S+")
RE_NOT = re.compile('[^\w\s@:,;]+')


def datareader(path: Path):
    with open(path, "rt", encoding="utf-8", errors='ignore') as fp:
        for row in csv.reader(fp):
            yield RE_NOT.sub(' ', RE_MENTION.sub('@', RE_URL.sub('', row[5]))), row[0]


def cleanup(src, target):
    """Nettoyage du jeu de tweet"""
    if not target.is_file():
        logging.info("Creating the text data file from %s", src)
        target_tmp = target.with_suffix(".tmp")
        with target_tmp.open("wt", encoding="utf-8") as out:
            for tweet, klass in datareader(src):
                out.write(tweet)
                out.write("\n")

        shutil.move(target_tmp, target)


Batch = namedtuple("Batch", ["text", "labels"])


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]:self.sizes[index + 1]], self.labels[index].item()

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels))


# --- Not used ??
# def process(mode: str, ds: CSVData, map: dict):
#     """Process the dataset
#     """
#     datapath = MAINDIR / f"{mode}.pth"
#     if datapath.is_file():
#         logging.info("Loading %s", mode)
#         with gzip.open(datapath, "rb") as fp:
#             return torch.load(fp)
#
#     text = array.array('L')
#     sizes = array.array('L')
#     labels = array.array('B')
#     sizes.append(0)
#     for tweet, label in tqdm(datareader(ds.files[mode]), unit=" sentences"):
#         for tokenid in tokenizer.encode_as_ids(tweet):
#             text.append(tokenid)
#         sizes.append(len(text))
#         labels.append(int(label))
#
#     data = TextDataset(torch.LongTensor(text), torch.LongTensor(sizes), torch.LongTensor(labels))
#     with gzip.open(datapath, "wb") as fp:
#         torch.save(data, fp)
#     return data


def generate_data(mode: str, tokenizer, vocab_size: int, data: CSVData, map):
    data_path = PROJECT_DIR / f"models/{mode}-{vocab_size}.pth"
    if data_path.is_file():
        return

    text = array.array('L')
    sizes = array.array('L')
    labels = array.array('B')
    sizes.append(0)
    for tweet, label in tqdm(datareader(data.path), unit=" sentences"):
        for tokenid in tokenizer.encode_as_ids(tweet):
            text.append(tokenid)
        label = int(label)
        if label in map:
            sizes.append(len(text))
            labels.append(map[label])

    data = TextDataset(torch.LongTensor(text), torch.LongTensor(sizes), torch.LongTensor(labels))
    with gzip.open(data_path, "wb") as fp:
        torch.save(data, fp)


@click.option("--vocab-size", default=1000, type=int, help="Vocabulary size")
@click.command()
def cli(vocab_size: int):
    # Création du jeu de données et du modèle
    ds = prepare_dataset("com.sentiment140.english")

    # Création du vocabulaire
    wpmodel = PROJECT_DIR / f"wp{vocab_size}.model"
    if not wpmodel.is_file():
        logging.info("Did not find the wordpiece model %s", wpmodel)
        TRAINPATH = Path("sentiment140-train.txt")
        cleanup(ds.train.path, TRAINPATH)
        logging.info("Création du vocabulaire avec sentencepiece")
        spm.SentencePieceTrainer.train(
            input=str(TRAINPATH),
            model_prefix=f"wp{vocab_size}",
            vocab_size=vocab_size
        )
        TRAINPATH.unlink()

    # Création des jeux de données
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(PROJECT_DIR / f"models/wp{vocab_size}.model"))

    CLASSMAP = {0: 0, 4: 1}
    logging.info("Traitement du train/test (Sentiment 140)")
    generate_data("test", tokenizer, vocab_size, ds.test, CLASSMAP)
    generate_data("train", tokenizer, vocab_size, ds.train, CLASSMAP)


if __name__ == "__main__":
    cli()
