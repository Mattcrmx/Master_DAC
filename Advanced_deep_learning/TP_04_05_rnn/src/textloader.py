from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
from typing import List
import torch
import re
PAD_IX = 0
EOS_IX = 1

LETTRES = string.ascii_lowercase + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES) + 2), LETTRES))
id2lettre[PAD_IX] = ''  ##NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return [lettre2id[c] for c in normalizeString(s)]


def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        self.text = text
        self.splitted_text = [a.strip() for a in self.text.split('.')]
        self.maxsent = maxsent
        self.maxlen = maxlen

    def __len__(self):
        return len(self.splitted_text)

    def __getitem__(self, i):
        return string2code(self.splitted_text[i])[:self.maxlen]


def collate_fn(samples: List[List[int]]):
    processed_samples = []
    max_len = max([len(sample) for sample in samples])

    for sample in samples:

        sample.append(EOS_IX)
        while len(sample) < max_len+1:
            sample.append(PAD_IX)

        processed_samples.append(sample)

    return torch.tensor(processed_samples)


if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=3)
    data = next(iter(loader))

    # Longueur maximum
    assert data.shape == (6, 3)

    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    # les chaînes sont identiques
    assert normalizeString(test) == " ".join([code2string(s).replace("|", " .") for s in data.t()])
