import datetime
import glob
import logging
import re
import os
import shutil
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamaestro import prepare_dataset
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
PAD_ID = 400001


class MLP(nn.Module):
    def __init__(self, dim_in, hidden_sizes, dim_out):
        super().__init__()
        sizes = [dim_in] + hidden_sizes
        layers = []

        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.ReLU()]
        layers += [nn.Linear(sizes[-1], dim_out)]  # last layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BaseModel(pl.LightningModule):

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, alpha = self(x)

        if self.global_step % 100 == 0:
            self.logger.experiment.add_histogram('entropy_output', Categorical(probs=alpha[:, :, 0]).entropy(),
                                                 self.global_step)

        loss = F.cross_entropy(logits, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, alpha = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)

        # log an html file with attention visualization
        # with open(f'{log_dir}/attention_{self.global_step}.html', 'w') as f:
        #     html_code = log_text_with_attention(x[0], alpha[0, :, 0])
        #     f.write(html_code)
        return loss


class LinearModel(BaseModel):
    def __init__(self, weights_embeddings: np.array, dim_h: int, lr: float, num_classes):
        super(LinearModel, self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)  # (vocab_size, dim_emb), including OOV/PAD tokens
        _, dim_emb = weights_embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(weights_embeddings)
        self.mlp = nn.Sequential(
            nn.Linear(dim_emb, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, num_classes)
        )
        self.lr = lr

    def forward(self, x: torch.LongTensor):
        x = self.embedding(x)  # (N, L, dim_emb)
        x = x.mean(dim=1)  # (N, dim_emb)
        logits = self.mlp(x)  # (N, num_classes)
        return logits, None


class SimpleAttentionModel(BaseModel):
    def __init__(self, weights_embeddings, num_classes, dim_h, lr):
        super(SimpleAttentionModel, self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)  # (vocab_size, dim_emb), including OOV/PAD tokens
        _, dim_emb = weights_embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(weights_embeddings)
        self.mlp = nn.Sequential(
            nn.Linear(dim_emb, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, num_classes)
        )
        self.q = nn.Linear(dim_emb, 1, bias=False)
        self.lr = lr

    def forward(self, x: torch.LongTensor):
        x_emb = self.embedding(x)  # (N, L, dim_emb)
        alpha = torch.where((x == PAD_ID).unsqueeze(dim=2), torch.tensor(-np.inf, dtype=torch.float),
                            self.q(x_emb))  # (N, L, 1)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.expand(x_emb.shape)  # (N, L, dim_emb)
        z = (alpha * x_emb).sum(dim=1)
        logits = self.mlp(z)  # (N, num_classes)
        return logits, alpha


class ComplexAttentionModel(BaseModel):
    def __init__(self, weights_embeddings, num_classes, dim_h, n_layers, lr, v_net, k_net):
        super(ComplexAttentionModel, self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)  # (vocab_size, dim_emb), including OOV/PAD tokens
        _, dim_emb = weights_embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(weights_embeddings)
        self.linear_q = MLP(dim_in=dim_emb, hidden_sizes=[dim_h] * n_layers, dim_out=dim_emb)
        self.v_net = v_net
        self.k_net = k_net
        if v_net:
            self.linear_v = MLP(dim_in=dim_emb, hidden_sizes=[dim_h] * n_layers, dim_out=dim_emb)
        if k_net:
            self.linear_k = MLP(dim_in=dim_emb, hidden_sizes=[dim_h] * n_layers, dim_out=dim_emb)
        self.mlp = MLP(dim_in=dim_emb, hidden_sizes=[dim_h, dim_h], dim_out=num_classes)
        self.lr = lr

    def forward(self, x: torch.LongTensor):
        x_emb = self.embedding(x)  # (N, L, dim_emb)

        if self.k_net:
            k = self.linear_k(x_emb)
        else:
            k = x_emb

        if self.v_net:
            v = self.linear_v(x_emb)
        else:
            v = x_emb

        # question
        N, L, d = k.shape
        q = self.linear_q(k.mean(dim=1))
        # batch dot product, from (N, L, 1, 1) to (N, L) after squeezing
        q_dot_x = torch.matmul(k.view(N, L, 1, d), q.view(N, 1, d, 1).repeat(1, L, 1, 1)).squeeze()

        # attention
        inf_tensor = torch.tensor(-np.inf, dtype=torch.float).type_as(
            q_dot_x)  # put the tensor on the right device (gpu if needed)
        alpha = torch.where((x == PAD_ID), inf_tensor, q_dot_x)  # (N, L)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(dim=2).expand(k.shape)  # (N, L, dim_emb)

        z = (alpha * v).sum(dim=1)
        logits = self.mlp(z)  # (N, num_classes)
        return logits, alpha


class SelfAttentionModel(BaseModel):
    def __init__(self, weights_embeddings, num_classes, dim_h, d, n_layers, lr):
        super().__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)  # (vocab_size, dim_emb), including OOV/PAD tokens
        _, dim_emb = weights_embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(weights_embeddings)
        self.q_list = [nn.Linear(dim_emb, d, bias=False) for i in range(n_layers)]
        self.k_list = [nn.Linear(dim_emb, d, bias=False) for i in range(n_layers)]
        self.v_list = [nn.Linear(dim_emb, dim_emb, bias=False) for i in range(n_layers)]
        self.g_list = [nn.Sequential(nn.Linear(dim_emb, dim_emb), nn.ReLU()) for i in range(n_layers)]
        self.lr = lr
        self.n_layers = n_layers
        self.d = d
        self.mlp = MLP(dim_in=dim_emb, hidden_sizes=[dim_h, dim_h], dim_out=num_classes)

    def forward(self, x: torch.LongTensor):
        x_emb = self.embedding(x)  # (N, L, dim_emb)
        x_l = x_emb
        attention_list = []
        for layer in range(self.n_layers):
            Q = self.q_list[layer](x_l)  # (N, L, d)
            K = self.k_list[layer](x_l)  # (N, L, d)
            V = self.v_list[layer](x_l)  # (N, L, dim_emb)

            logits = (Q @ K) / np.sqrt(self.d)  # (N, L, L)
            inf_tensor = torch.tensor(-np.inf, dtype=torch.float).type_as(
                logits)  # put the tensor on the right device (gpu if needed)
            padded_indices = (x @ x.T == PAD_ID)
            alpha = torch.where(padded_indices, inf_tensor, logits)  # (N, L, L)
            alpha = F.softmax(alpha, dim=2)
            x_self_attention = alpha @ V  # (N, L, dim_emb)
            x_l = self.g_list[layer](x_l + x_self_attention)
            attention_list.append(alpha)
        representation = x_l.mean(dim=1)
        return self.mlp(representation), attention_list


def log_text_with_attention(sentence, scores):
    sentence = sentence.tolist()
    scores = scores.tolist()
    tokens = [id2word[x] for x in sentence]
    cmap = cm.get_cmap('BuGn')

    off = (sum(scores) / len(scores))
    normer = matplotlib.colors.Normalize(vmin=min(scores) - off, vmax=max(scores) + off)
    colors = [matplotlib.colors.to_hex(cmap(normer(x))) for x in scores]

    style_elems = []
    span_elems = []
    for i in range(len(tokens)):
        style_elems.append(f'.c{i} {{ background-color: {colors[i]}; }}')
        span_elems.append(f'<span class="c{i}">{tokens[i]} </span>')

    return f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet"><style>span {{ font-family: "Roboto Mono", monospace; font-size: 12px; }} {' '.join(style_elems)}</style></head><body>{' '.join(span_elems)}</body></html>"""


def collate_fn(batch):
    x, y = zip(*batch)
    return pad_sequence([torch.tensor(s) for s in x], padding_value=PAD_ID, batch_first=True), torch.tensor(y)


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    assert PAD_ID == len(words) + 1
    words.append('__OOV__')
    words.append('__PAD__')

    word2id = {word: ix for ix, word in enumerate(words)}
    id2word = {ix: word for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size), np.zeros(embedding_size)))  # zero columns for OOV/PAD

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, id2word, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(
        ds.test.classes, ds.test.path, tokenizer, load=False)


def save_src_and_config():
    # save config and source files as text files
    filename = PROJECT_DIR / log_dir / 'conf.yaml'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        OmegaConf.save(conf, f)
    for f in glob.iglob('*.py'):
        shutil.copy2(f, log_dir)

    print(f'Running with conf: \n{OmegaConf.to_yaml(conf)}')
    conf_clean = {k: str(v) for (k, v) in conf.items()}
    logger.experiment.add_hparams(conf_clean, metric_dict={'score': 0.})


if __name__ == '__main__':
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('conf.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = f'runs/{conf.model_type}-dim_h{conf.dim_h}-layers-{conf.n_layers}' \
              f'-use_v{conf.v_net}-use_k{conf.k_net}-{time_tag}'
    logger = TensorBoardLogger(log_dir)
    save_src_and_config()

    word2id, id2word, embeddings, train_dataset, val_dataset = get_imdb_data(embedding_size=conf.dim_emb)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=conf.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=conf.batch_size)

    if conf.model_type == 'linear':
        model = LinearModel(weights_embeddings=embeddings, dim_h=conf.dim_h, lr=conf.lr, num_classes=2)
    elif conf.model_type == 'simple_attention':
        model = SimpleAttentionModel(weights_embeddings=embeddings, dim_h=conf.dim_h, lr=conf.lr, num_classes=2)
    elif conf.model_type == 'complex_attention':
        model = ComplexAttentionModel(weights_embeddings=embeddings, dim_h=conf.dim_h, n_layers=conf.n_layers,
                                      lr=conf.lr, num_classes=2, v_net=conf.v_net, k_net=conf.k_net)
    elif conf.model_type == 'self_attention':
        # todo : debug implementation
        model = SelfAttentionModel(weights_embeddings=embeddings, dim_h=conf.dim_h, d=500, n_layers=conf.n_layers,
                                   lr=conf.lr, num_classes=2)
    else:
        raise ValueError

    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
