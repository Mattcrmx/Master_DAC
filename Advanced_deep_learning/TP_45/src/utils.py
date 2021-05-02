import csv
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)


def masked_cross_entropy(output: torch.Tensor, target: torch.Tensor):
    """

    Args:
        output: Output logits tensor of shape (L, N, num_classes)
        target: Target indices tensor of shape (L, N)

    Returns:

    """
    # The cross entropy loss expects a logit tensor of shape (N, num_classes, d) and target tensor
    # of shape (N, d) where d is any additional dimension (here, d=L)
    output = output.permute(1, 2, 0)
    target = target.t()
    loss = F.cross_entropy(output, target, reduction='none')

    # Only consider valid characters (by replacing the loss with zero for PAD target characters)
    is_valid_character = (target != 0).detach()
    loss = loss * is_valid_character

    n_valid = is_valid_character.sum()
    return loss.sum() / n_valid


def fill_na(mat):
    ix, iy = np.where(np.isnan(mat))
    for i, j in zip(ix, iy):
        if np.isnan(mat[i + 1, j]):
            mat[i, j] = mat[i - 1, j]
        else:
            mat[i, j] = (mat[i - 1, j] + mat[i + 1, j]) / 2.
    return mat


def read_temps(path):
    """Lit le fichier de températures"""
    data = []
    with open(path, "rt") as fp:
        reader = csv.reader(fp, delimiter=',')
        next(reader)
        for row in reader:
            data.append([float(x) if x != "" else float('nan') for x in row[1:]])
    return torch.tensor(fill_na(np.array(data)), dtype=torch.float32)


class RNN(nn.Module):
    def __init__(self, input_dim, latent, output_dim, encode, decode, batch_size, use_embedding=False):
        super().__init__()
        self.latent = latent
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.sigma_encode = encode
        self.sigma_decode = decode
        self.use_embedding = use_embedding
        # encode layer
        if use_embedding:
            # for text generation, use an embedding to avoid encoding input as one hot
            self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=latent, padding_idx=0)
        else:
            self.w_i = torch.nn.Parameter(torch.zeros((input_dim, latent), dtype=torch.float32))
            torch.nn.init.xavier_uniform_(self.w_i)
        self.w_h = torch.nn.Parameter(torch.empty((latent, latent), dtype=torch.float32))
        self.b_h = torch.nn.Parameter(torch.zeros((latent,), dtype=torch.float32))
        # decode layer
        self.w_d = torch.nn.Parameter(torch.empty((latent, output_dim), dtype=torch.float32))
        self.b_d = torch.nn.Parameter(torch.zeros((output_dim,), dtype=torch.float32))

        # weights initialization
        torch.nn.init.xavier_uniform_(self.w_h)
        torch.nn.init.xavier_uniform_(self.w_d)

    def one_step(self, x, h_current):
        if self.use_embedding:
            x = self.embedding(x)
        else:
            x = x @ self.w_i
        return self.sigma_encode(x + h_current @ self.w_h + self.b_h)

    def forward(self, x, device='cpu'):
        """

        Args:
            x: Input tensor of shape (L, N, input_dim) or (L, N) if not one-hot encoded
            device:

        Returns:
            The tensor h of hidden states, of dimension (L, N, h_dim)
        """
        if self.use_embedding:
            # input is a tensor of indexes, not one-hot encoded
            seq_length, batch_size = x.shape
        else:
            seq_length, batch_size, _ = x.shape

        h_current = torch.zeros(batch_size, self.latent, device=device)
        h = []
        # h_current = h_current.type_as(x)
        # h = h.type_as(x)
        for t in range(seq_length):
            h_current = self.one_step(x[t], h_current)
            h.append(h_current)
        return torch.stack(h)

    def decode(self, h):
        """

        Args:
            h: Either a tensor of shape (T, N, h_dim) if h is the whole sequence,
                or a tensor of shape (N, h_dim) if h is simply h_t

        Returns:
            y, a tensor of shape (T, N, out_dim) or (N, out_dim) (depending on the shape of h)
        """
        d = len(h.shape)
        assert d == 2 or d == 3

        if self.sigma_decode:
            return self.sigma_decode(h @ self.w_d + self.b_d, dim=d - 1)
        return h @ self.w_d + self.b_d


class GRU(nn.Module):
    def __init__(self, dim_input, dim_embedding, dim_hidden, dim_output):
        super().__init__()
        self.dim_hidden = dim_hidden

        # from input tensor of indices to embedded vector
        self.embedding = nn.Embedding(dim_input, dim_embedding)
        # dim_z = dim_r = dim_hidden
        self.linear_z = nn.Linear(dim_hidden + dim_embedding, dim_hidden)
        self.linear_r = nn.Linear(dim_hidden + dim_embedding, dim_hidden)
        self.linear_h = nn.Linear(dim_hidden + dim_embedding, dim_hidden)

        self.linear_y = nn.Linear(dim_hidden, dim_output)

    def one_step(self, x, h_current):
        x = self.embedding(x)
        hx_concat = torch.cat([h_current, x], dim=1)

        # forget gate
        z = torch.sigmoid(self.linear_z(hx_concat))
        # reset gate
        r = torch.sigmoid(self.linear_r(hx_concat))

        h_tilde = torch.tanh(self.linear_h(
            torch.cat([r * h_current, x], dim=1)
        ))
        h_new = (1. - z) * h_current + z * h_tilde
        return h_new

    def forward(self, x):
        """

        Args:
            x: Input tensor of indices, of shape (L, N)

        Returns:
            The tensor h of hidden states, of dimension (L, N, dim_h)
        """
        seq_length, batch_size = x.shape
        h_current = torch.zeros(batch_size, self.dim_hidden)
        h = []
        for t in range(seq_length):
            h_current = self.one_step(x[t], h_current)
            h.append(h_current)
        return torch.stack(h)

    def decode(self, h):
        """

        Args:
            h: Either a tensor of shape (T, N, h_dim) if h is the whole sequence,
                or a tensor of shape (N, h_dim) if h is simply h_t

        Returns:
            y, a tensor of shape (T, N, out_dim) or (N, out_dim) (depending on the shape of h)
        """
        return self.linear_y(h)


class LSTM(nn.Module):
    def __init__(self, dim_input, dim_embedding, dim_hidden, dim_output):
        super().__init__()
        self.dim_hidden = dim_hidden

        # from input tensor of indices to embedded vector
        self.embedding = nn.Embedding(dim_input, dim_embedding)
        # dim_z = dim_r = dim_hidden
        self.linear_i = nn.Linear(dim_hidden + dim_embedding, dim_hidden)
        self.linear_f = nn.Linear(dim_hidden + dim_embedding, dim_hidden)
        self.linear_c = nn.Linear(dim_hidden + dim_embedding, dim_hidden)
        self.linear_o = nn.Linear(dim_hidden + dim_embedding, dim_hidden)

        self.linear_y = nn.Linear(dim_hidden, dim_output)

    def one_step(self, x, h_current, c_current):
        x = self.embedding(x)
        hx_concat = torch.cat([h_current, x], dim=1)

        # forget gate
        f = torch.sigmoid(self.linear_f(hx_concat))
        # input gate
        i = torch.sigmoid(self.linear_i(hx_concat))
        # internal memory
        c_new = f * c_current + i * torch.tanh(self.linear_c(hx_concat))
        # output gate
        o = torch.sigmoid(self.linear_o(hx_concat))
        # new hidden state
        h_new = o * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, return_last_internal=False):
        """

        Args:
            x: Input tensor of indices, of shape (L, N)
            return_last_internal:

        Returns:
            The tensor h of hidden states, of dimension (L, N, dim_h)
            And optionally, c_L the last internal memory of dim (N, dim_h)
        """
        seq_length, batch_size = x.shape
        h_current = torch.zeros(batch_size, self.dim_hidden)
        c_current = torch.zeros(batch_size, self.dim_hidden)
        h = []
        for t in range(seq_length):
            h_current, c_current = self.one_step(x[t], h_current, c_current)
            h.append(h_current)

        if return_last_internal:
            return torch.stack(h), c_current
        return torch.stack(h)

    def decode(self, h):
        """

        Args:
            h: Either a tensor of shape (T, N, h_dim) if h is the whole sequence,
                or a tensor of shape (N, h_dim) if h is simply h_t

        Returns:
            y, a tensor of shape (T, N, out_dim) or (N, out_dim) (depending on the shape of h)
        """
        return self.linear_y(h)


class LightningRNN(pl.LightningModule):
    def __init__(self, hparams):
        """
        Base Lightning module for text generation, in charge of configuring the optimizers and logging the weights/grad
        histograms. This class has to be overriden to define at least a model and a training/validation step,
        depending on the task.

        Args:
            hparams:
        """
        super(LightningRNN, self).__init__()
        self.hparams = hparams
        if self.hparams.criterion == 'CE':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.hparams.criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.hparams.criterion == 'maskedCE':
            self.criterion = masked_cross_entropy

        # encode and decode activation functions
        # todo: remove for simplicity ! (+ we only use tanh in the end)
        encode_dict = {'tan_h': torch.tanh,
                       'relu': F.relu,
                       'sigmoid': F.sigmoid,
                       'leaky_relu': F.leaky_relu}
        decode_dict = {'softmax': F.softmax}
        self.sigma_encode = encode_dict.get(hparams.encode, None)
        self.sigma_decode = decode_dict.get(hparams.decode, None)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'SGD':
            return torch.optim.SGD(params=self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'adam':
            return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f'optimizer {self.hparams.optimizer} not valid')

    def on_after_backward(self):
        global_step = self.global_step
        if global_step % self.hparams.log_freq_grad == 0:
            for name, param in self.model.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)


class TempDatasetClassif(Dataset):
    def __init__(self, data, max_length):
        self.max_length = max_length
        self.data = data.unsqueeze(2)
        T, n, _ = self.data.shape
        self.labels = torch.tensor([idx // (T - self.max_length) for idx in range(len(self))])

    def __len__(self):
        T, n, _ = self.data.shape
        return (T - self.max_length) * n

    def __getitem__(self, idx):
        # todo: variable lengths batches
        T, n, _ = self.data.shape
        t = idx % (T - self.max_length)
        city = idx // (T - self.max_length)
        return self.data[t:t + self.max_length, city], self.labels[idx]


class TempDatasetForecast(Dataset):
    def __init__(self, data, seq_length, predict_window):
        self.seq_length = seq_length
        self.predict_window = predict_window
        self.data = data.unsqueeze(2)

        T, n, _ = self.data.shape
        self.labels = torch.tensor([idx // (T - self.seq_length) for idx in range(len(self))])

    def __len__(self):
        T, n, _ = self.data.shape
        return (T - self.seq_length - self.predict_window) * n

    def __getitem__(self, idx):
        T, n, _ = self.data.shape
        t = idx % (T - self.seq_length - self.predict_window)
        city = idx // (T - self.seq_length - self.predict_window)

        x = self.data[t: t + self.seq_length, city]
        y = self.data[t + self.seq_length: t + self.seq_length + self.predict_window, city]
        return x, y, t, city


class SpeechDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        Dataset used by `exo4.py` for a first approach to character-level text generation (starting with only the
         first character).

        Another dataset will be used in TP5 (`tp5.py`), where we use full sentences,
        several characters as a starting sequence and we use curriculum learning
        (alternate between network predictions and ground truth).

        Args:
            data: Tensor of shape (L,) with encoded text (not as one-hot)
            seq_length: Fixed length of the target sequences (using 1 character as a starting point)
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        L, = self.data.shape
        return L // self.seq_length

    def __getitem__(self, idx):
        return self.data[idx * self.seq_length: (idx + 1) * self.seq_length]
