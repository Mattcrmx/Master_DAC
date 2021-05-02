import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm


# --- Modules
class AutoEncoder(torch.nn.Module):
    def __init__(self, D_in, H, tie_weights=False):
        super(AutoEncoder, self).__init__()
        self.tie_weights = tie_weights

        # Encoder part
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU()
        )

        if tie_weights:
            # Tie the weights of the encoder and the decoder part, for easier training
            # (but may not be optimal if highly nonlinear data)
            self.b2 = torch.nn.Parameter(torch.randn(D_in), requires_grad=True)
        else:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(H, D_in),
                torch.nn.Sigmoid()
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if self.tie_weights:
            W = self.encoder[0].weight
            return torch.sigmoid(F.linear(z, weight=W.t(), bias=self.b2))
        else:
            return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class Highway(nn.Module):
    def __init__(self, D_in, H, D_out, num_layers):
        super(Highway, self).__init__()

        # Linear layer
        self.layer1 = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU()
        )

        # Highway modules
        self.num_layers = num_layers
        self.linear_h = nn.ModuleList([nn.Linear(H, H) for _ in range(num_layers)])
        self.linear_x = nn.ModuleList([nn.Linear(H, H) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(H, H) for _ in range(num_layers)])
        self.f = torch.nn.ReLU()  # activation function

        # Classification layer
        self.last_layer = nn.Sequential(
            nn.Linear(H, D_out),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Implements a Highway network in the middle of our classification model, like in the
        original paper (https://arxiv.org/abs/1505.00387):
        ”applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid and ⨀ is
        element-wise multiplication”

        Args:
            x: tensor with shape (N, D_in)

        Returns:
            y: tensor with logits of shape (N, D_out)

        """
        x = self.layer1(x)

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.linear_h[layer](x))
            linear = self.linear_x[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        y = self.last_layer(x)
        return y


# --- Utils
class MNIST(Dataset):
    def __init__(self, images, labels, device):
        # copy the np arrays to remove the UserWarning, as they are not writeable
        self.images = torch.tensor(images.copy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.copy(), dtype=torch.int64)
        self.images /= 255.0

        _, self.width, self.height = self.images.shape
        self.n_features = self.width * self.height

        # Use GPU if available
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AE',
                        help="'AE' for AutoEncoder, 'highway' for the highway network")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--criterion', type=str, default='cross-entropy',
                        help="AE: 'mse' for mean squared error, 'bce' for binary cross entropy\n"
                             "highway: 'cross-entropy' for cross entropy")
    parser.add_argument('--optim', type=str, default='sgd',
                        help="'sgd' or 'adam'")
    # AE specific parameters
    parser.add_argument('--ae_dim', type=int, default=100,
                        help='Size of the latent space of the AutoEncoder')
    parser.add_argument('--tie_weights', action='store_true')
    # Highway network parameters
    parser.add_argument('--highway_dim', type=int, default=50)
    parser.add_argument('--highway_num', type=int, default=2)
    args = parser.parse_args()

    # savepath = Path("model.pch") todo
    writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # checks if gpu available

    # --- Prepare data
    ds = prepare_dataset("com.lecun.mnist")
    train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
    train_dataset = MNIST(train_images, train_labels, device=device)
    test_dataset = MNIST(test_images, test_labels, device=device)
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # --- Create model and optimizer
    if args.criterion == 'mse':
        criterion = torch.nn.MSELoss()
    elif args.criterion == 'bce':
        criterion = torch.nn.BCELoss()
    elif args.criterion == 'cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Criterion {args.criterion} not implemented')
    if args.model == 'AE':
        model = AutoEncoder(D_in=train_dataset.n_features, H=args.ae_dim, tie_weights=args.tie_weights)
    elif args.model == 'highway':
        model = Highway(D_in=train_dataset.n_features, H=args.highway_dim, D_out=10, num_layers=args.highway_num)
    else:
        raise ValueError(f'Model {args.model} not implemented')
    model.to(device)
    if args.optim == 'sgd':
        optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Optimizer {args.optim} not implemented')

    # --- Train model
    print(f'Training...')
    for ep in tqdm(range(args.max_epochs)):
        epoch_loss = []
        for train_sample, train_label in data_loader:
            # Set the gradients to zero
            optim.zero_grad()

            # forward, backward and optimize
            train_sample = train_sample.reshape(-1, train_dataset.n_features)
            y_hat = model(train_sample)
            if args.model == 'AE':
                loss = criterion(y_hat, train_sample)
            else:
                loss = criterion(y_hat, train_label)
            loss.backward()
            optim.step()

            epoch_loss.append(loss.item())

        with torch.no_grad():
            # estimate train loss with the saved epoch values
            train_loss = np.mean(epoch_loss)
            # compute test loss on the test dataset
            x_test = test_dataset.images.reshape(-1, train_dataset.n_features)
            y_test = test_dataset.labels
            x_test_predict = model(x_test)
            if args.model == 'AE':
                test_loss = criterion(x_test_predict, x_test).item()
            else:
                test_loss = criterion(x_test_predict, y_test).item()
                predicted = torch.argmax(x_test_predict, dim=1)
                test_accuracy = (predicted == y_test).float().sum().item() / len(test_dataset)

            # log values to tensorboard
            writer.add_scalar(f'{args.criterion}/train', train_loss, ep)
            writer.add_scalar(f'{args.criterion}/test', test_loss, ep)

            if args.model == 'highway':
                writer.add_scalar(f'accuracy/test', test_accuracy, ep)
            if ep % 5 == 0 and args.model == 'AE':
                # save images to tensorboard
                images_original = make_grid(x_test[:32].reshape(-1, 28, 28).unsqueeze(1).repeat(1, 3, 1, 1))
                images_predict = make_grid(x_test_predict[:32].reshape(-1, 28, 28).unsqueeze(1).repeat(1, 3, 1, 1))
                writer.add_images('images', torch.stack((images_original, images_predict)), ep)

    metrics = {f'hparam/{args.criterion}': test_loss}
    if args.model == 'highway':
        metrics['hparam/accuracy'] = test_accuracy
    print(f'Training done, final score: {metrics}')
    writer.add_hparams(vars(args), metric_dict=metrics)
    writer.close()
