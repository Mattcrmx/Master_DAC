import datetime
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
from typing import List
import torch
import re
from utils import LightningRNN, RNN, LSTM, GRU
import numpy as np

PAD_IX = 0
EOS_IX = 1

LETTRES = string.ascii_letters + string.digits + '.!?' + ' '
id2lettre = dict(zip(range(2, len(LETTRES) + 2), LETTRES))
id2lettre[PAD_IX] = ''  # NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if c in LETTRES
    )


def normalize_string(s):
    # Trim and remove non-letter characters
    s = re.sub("[^a-zA-Z0-9.!?']+", " ", s)
    s = unicode_to_ascii(s.strip())
    return s


def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return [lettre2id[c] for c in normalize_string(s)]


def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


def collate_fn(samples: List[List[int]]):
    processed_samples = []
    max_len = max([len(sample) for sample in samples])

    for sample in samples:
        sample.append(EOS_IX)
        while len(sample) < max_len + 1:
            sample.append(PAD_IX)
        processed_samples.append(sample)

    return torch.tensor(processed_samples)


class TextDataset(Dataset):
    def __init__(self, text: str, max_len=None):
        """

        Args:
            text:
            max_len:
        """
        self.text = text
        self.splitted_text = [a.strip() for a in self.text.split('.')]
        self.max_len = max_len

    def __len__(self):
        return len(self.splitted_text)

    def __getitem__(self, i):
        return string2code(self.splitted_text[i])[:self.max_len]


class MaskedAccuracy(pl.metrics.Accuracy):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: (L, N, out_dim) or (L, N) Predictions from model (logits or values)
            target: (L, N) Ground truth values
        """
        if len(preds.shape) == 2:
            preds = preds.t()  # now (N, L)
        elif len(preds.shape) == 3:
            preds = preds.permute(1, 2, 0)  # now (N, out_dim, L)
            preds = torch.argmax(preds, dim=1)  # (N, L)
        target = target.t()  # now (N, L)
        assert preds.shape == target.shape

        is_valid_character = (target != PAD_IX).detach()
        self.correct += torch.sum((preds == target) * is_valid_character)
        self.total += target.numel()


class TextGenerationRNN(LightningRNN):
    def __init__(self, alphabet_size, code2string, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = self.hparams.seq_length
        self.alphabet_size = alphabet_size
        self.code2string = code2string
        # our model takes the previous one hot encoded character, and outputs the logits of the next character
        if self.hparams.model_type == 'rnn':
            self.model = RNN(input_dim=alphabet_size, latent=self.hparams.latent_dim, output_dim=alphabet_size,
                             encode=self.sigma_encode, decode=None, batch_size=self.hparams.batch_size,
                             use_embedding=True)
        elif self.hparams.model_type == 'lstm':
            self.model = LSTM(dim_input=alphabet_size, dim_embedding=self.hparams.dim_embedding,
                              dim_hidden=self.hparams.latent_dim, dim_output=alphabet_size)
        elif self.hparams.model_type == 'gru':
            self.model = GRU(dim_input=alphabet_size, dim_embedding=self.hparams.dim_embedding,
                             dim_hidden=self.hparams.latent_dim, dim_output=alphabet_size)
        else:
            raise ValueError(f'Model type {self.hparams.model_type} not implemented')

        self.train_acc = MaskedAccuracy()
        self.val_acc_argmax = MaskedAccuracy()
        self.val_acc_beam = MaskedAccuracy()

    def forward(self, x_start, x_next, teacher_forcing_ratio):
        """

        Args:
            x_start: (start_len, N)
            x_next: (L - start_len, N)
            teacher_forcing_ratio:

        Returns:

        """
        predict_len, _ = x_next.shape  # length of the rest of the sentence, that we try to predict

        # don't train on the first characters
        with torch.no_grad():
            if self.hparams.model_type == 'lstm':
                h, c_prev = self.model(x_start, return_last_internal=True)
            else:
                h = self.model(x_start)
            h_prev = h[-1]  # (N, dim_h)

        logits = []
        for t in range(predict_len):
            # get next character logits
            logits_t = self.model.decode(h_prev)  # (N, out_dim)
            logits.append(logits_t)

            # use either the predicted character or the ground truth as the next decoder input
            teacher_force = np.random.random() < teacher_forcing_ratio
            if teacher_force:
                x_t = x_next[t]  # (N,)
            else:
                x_t = logits_t.argmax(dim=1)  # (N,)

            # get the next hidden state
            if self.hparams.model_type == 'lstm':
                h_prev, c_prev = self.model.one_step(x_t, h_prev, c_prev)
            else:
                h_prev = self.model.one_step(x_t, h_prev)

        logits = torch.stack(logits, dim=0)  # (predict_len, N, out_dim)
        return logits

    def training_step(self, batch, batch_idx):
        # --- Previous approaches:
        # - we tried using teacher forcing only (and always feeding the ground truth as an input)
        #   The network was able to lean something (loss was decreasing), but it was not very good
        #   at predictions
        # - we tried alternating teacher forcing with the network predictions, at the epoch level
        #   (taking either the ground truth or the beam search prediction)
        #   however this makes training more chaotic, and beam search is actually better used for inference,
        #   not for training
        # Our current approach now uses teacher forcing at each time step (taking either the ground truth or the
        # most probable character as an input), and we use beam_search for validation only.
        #
        #
        # # Progressively decrease teacher forcing
        # threshold = np.exp(-self.current_epoch / self.hparams.teacher_forcing_tau)
        # use_teacher_forcing = np.random.random() < threshold
        # if use_teacher_forcing:
        #     h = self.model(x)
        # else:
        #     # Instead of teacher forcing, use the beam search predictions
        #     x_beam_search = self.beam_search_decode(x, k=self.hparams.beam_search_k, start_len=start_len)
        #     h = self.model(x_beam_search)

        batch = batch.T  # (L, N) after transpose
        start_len = self.hparams.start_len

        x_start = batch[:start_len]
        y = batch[start_len:]  # (L - start_len, N)
        threshold = np.exp(-self.hparams.teacher_forcing_decay * self.current_epoch)
        y_logits = self(x_start=x_start, x_next=y, teacher_forcing_ratio=threshold)  # (L - start_len, N, out_dim)

        # our custom maskedCrossEntropy
        loss = self.criterion(y_logits, y)

        if self.global_step % self.hparams.log_freq_text == 0:
            # strictly speaking, our y_logits are not fully from the argmax, as we use teacher forcing
            # (at least at the beginning of training)
            self.log_text(x_start=x_start, y=y, logits_argmax=y_logits, tag='train')

        acc = self.train_acc(y_logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('teacher_forcing_ratio', threshold)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.T  # (L, N) after transpose
        start_len = self.hparams.start_len

        x_start = batch[:start_len]
        y = batch[start_len:]  # (L - start_len, N)
        logits_argmax = self(x_start=x_start, x_next=y, teacher_forcing_ratio=0)  # (L - start_len, N, out_dim)
        y_beam = None
        if self.hparams.model_type != 'lstm':
            y_beam = self.beam_search_decode(batch, k=self.hparams.beam_search_k, start_len=start_len,
                                             nucleus_sampling=self.hparams.nucleus_sampling)  # (L, N)
            acc_beam = self.val_acc_beam(y_beam[start_len:], y)
            self.log('val_acc_beam', acc_beam)

        self.log_text(x_start=x_start, y=y, logits_argmax=logits_argmax, y_beam=y_beam, tag='val')

        loss = self.criterion(logits_argmax, y)
        acc_argmax = self.val_acc_argmax(logits_argmax, y)
        self.log('val_loss', loss)
        self.log('val_acc_argmax', acc_argmax)
        return loss

    def log_text(self, x_start, y, logits_argmax, y_beam=None, tag=''):
        text_start = self.code2string(x_start[:, 0])
        text_label = self.code2string(y[:, 0])
        predicted = logits_argmax.argmax(dim=2)  # shape (predicted_len, N)
        text_argmax = self.code2string(predicted[:, 0])
        self.logger.experiment.add_text(f'{tag}_label', f'{text_start}... {text_label}', self.global_step)
        self.logger.experiment.add_text(f'{tag}_argmax', f'{text_start}... {text_argmax}', self.global_step)
        if y_beam is not None:
            start_len = len(text_start)
            text_beam = self.code2string(y_beam[start_len:, 0])
            self.logger.experiment.add_text(f'{tag}_beam', f'{text_start}... {text_beam}', self.global_step)

    def get_next_top_k(self, h_prev, x_t, k):
        """

        Args:
            h_prev: previous hidden state h_(t-1), (N, h_dim)
            x_t: next character input (N,)
            k:

        Returns:
            h_t, top_logp, top_chars
        """
        h_t = self.model.one_step(x_t, h_prev)  # (N, h_dim)
        logp = torch.log_softmax(self.model.decode(h_t), dim=1)  # (N, alphabet_size)
        logp = logp.transpose(0, 1)  # (alphabet_size, N)
        top_logp, top_chars = torch.topk(logp, k, dim=0)  # (K, N)

        return h_t, top_logp, top_chars

    def beam_search_decode(self, x, k, start_len, nucleus_sampling):
        L, N = x.shape
        h_dim = self.hparams.latent_dim

        # First, get the encoding of the starting sequence
        h_start = self.model(x[:start_len - 1])[-1]
        # And get the first top k options, given the starting sequence
        h_t, top_logp, top_chars = self.get_next_top_k(
            h_prev=h_start,
            x_t=x[start_len], k=k
        )

        top_k_logp = top_logp  # (K, N)
        # Copy starting sequence and hidden state K times
        top_k_hidden_states = h_t.repeat(k, 1, 1)  # (K, N, h_dim)
        start_seq = x[:start_len].repeat(k, 1, 1)  # (K, start_char, N)
        top_k_seq = torch.cat([start_seq, top_chars.view(k, 1, N)], dim=1)  # (K, start_char + 1, N)

        # Now, predict the next characters (up to the sequence length L)
        for t in range(start_len + 1, L):
            top_kk_logp = []
            top_kk_chars = []
            h_list = []
            for i in range(k):
                h_t, top_logp, top_chars = self.get_next_top_k(
                    h_prev=top_k_hidden_states[i],
                    x_t=top_k_seq[i, -1, :], k=k
                )
                # get total sequence log likelihood, by adding the previous sequence log probas
                logp_prev = top_k_logp[i]  # (N,)
                if nucleus_sampling:
                    # only consider the k top characters in the probabilities
                    top_logp = torch.nn.functional.log_softmax(top_logp, dim=0)
                top_logp += logp_prev  # (K, N)

                h_list.append(h_t)
                top_kk_logp.append(top_logp)
                top_kk_chars.append(top_chars)

            # reshape tensors so we can easily select the top k options
            top_kk_logp = torch.cat(top_kk_logp, dim=0)  # (K * K, N)
            top_kk_chars = torch.cat(top_kk_chars, dim=0)  # (K * K, N)
            new_h = torch.cat(h_list, dim=0).view(1, k, N, h_dim)  # (1, K, N, h_dim)
            new_h = new_h.expand(k, k, N, h_dim).flatten(end_dim=1)  # (K * K, N, h_dim)

            # select the top k new characters among the k*k options
            new_logp, indices = torch.topk(top_kk_logp, k, dim=0)  # (K, N)
            new_chars = top_kk_chars.gather(0, indices)  # (K, N)
            new_h = new_h.gather(0, indices.view(k, N, 1).expand(k, N, h_dim))  # (K, N, h_dim)

            # update the tensors in memory
            top_k_logp = new_logp
            top_k_hidden_states = new_h
            top_k_seq = torch.cat([top_k_seq, new_chars.view(k, 1, N)], dim=1)

        # Finally, select and return the N top sequences
        top_i = top_logp.argmax(dim=0)  # (N,)
        top_seq = top_k_seq.gather(0, top_i.view(1, 1, N).expand(1, L, N))
        top_seq = top_seq.squeeze()  # (L, N)
        return top_seq


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config/tp5.yaml'))
    print(OmegaConf.to_yaml(conf))
    time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = TensorBoardLogger(f'../runs/tp5-{conf.model_type}-latent-{conf.latent_dim}'
                               f'-tf_decay-{conf.teacher_forcing_decay}-nuc_samp-{conf.nucleus_sampling}'
                               f'-{time_tag}')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=3,
        mode='min',
    )

    # --- data loading
    with open(PROJECT_DIR / 'data/trump_full_speech.txt') as f:
        data_str = f.read()
    data_str = ''.join([line for line in data_str])
    # Take only the first half of the dataset to speed up training
    # data_str = data_str[:int(0.1 * len(data_str))]
    # split into train and validation data
    size_train = int(0.7 * len(data_str))
    train_dataset = TextDataset(data_str[:size_train], max_len=conf.max_len)
    val_dataset = TextDataset(data_str[size_train:], max_len=conf.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, collate_fn=collate_fn, shuffle=True,
                                  drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, collate_fn=collate_fn, drop_last=True,
                                num_workers=4)

    rnn_pl = TextGenerationRNN(alphabet_size=len(id2lettre), code2string=code2string, hparams=conf)

    # train model
    trainer = pl.Trainer(**OmegaConf.to_container(conf.trainer), logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(rnn_pl, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
