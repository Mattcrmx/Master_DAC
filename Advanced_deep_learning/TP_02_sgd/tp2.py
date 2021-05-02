import torch
from torch.utils.tensorboard import SummaryWriter
import datamaestro
import argparse
import datetime
from functions import MSE, Linear, train_loop
from networks import MyNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--strategy', type=str, default='linear',
                        help="'linear' for a linear regression, 'neural_nets' for the neural nets")
    parser.add_argument('--model', type=str, default='handmade', help="'handmade' or 'sequential'")
    parser.add_argument('--criterion', type=str, default='mse', help="'mse', 'L1', 'cross_entropy'")
    parser.add_argument('--batch_size', type=int, default=1)  # SGD
    parser.add_argument('--activation', type=str, default='tan_h', help="'tan_h', 'ReLU', 'sigmoid', 'leaky_ReLU'")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.1)

    args = parser.parse_args()
    print(args.__dict__)

    writer = SummaryWriter(f'runs/lr_{args.lr}_batch_size{args.batch_size}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

    data = datamaestro.prepare_dataset("edu.uci.boston")
    col_names, data_x, data_y = data.data()
    data_x = torch.tensor(data_x, dtype=torch.float)
    data_y = torch.tensor(data_y, dtype=torch.float).reshape(-1, 1)
    Scaler = StandardScaler()

    # Splitting the data to perform the training
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=420)
    X_train_scaled = torch.tensor(Scaler.fit_transform(X_train)).float()
    X_test_scaled = torch.tensor(Scaler.fit_transform(X_test)).float()

    batch_size = args.batch_size

    if args.strategy == 'linear':
        # parameters linear regression

        w = torch.rand(X_train.size()[1], 1, requires_grad=True, dtype=torch.float32)
        b = torch.rand(1, requires_grad=True, dtype=torch.float32)
        eps = args.lr * batch_size
        mse = MSE.apply
        linear = Linear.apply

        # training loop linear regression
        train_loop(X_train_scaled, y_train, X_test_scaled, y_test, linear, mse, args.criterion, eps, batch_size, writer,
                   w=w, b=b)

    elif args.strategy == 'neural_nets':
        # parameters neural nets

        if args.model == 'handmade':
            model = MyNetwork(X_train_scaled.shape[1], 10, 1)

        elif args.model == 'sequential':
            activation_dict = {'tan_h': torch.nn.Tanh, 'ReLU': torch.nn.ReLU, 'sigmoid': torch.nn.Sigmoid,
                               'leaky_ReLU': torch.nn.LeakyReLU}

            model = torch.nn.Sequential(torch.nn.Linear(X_train_scaled.shape[1], 10),
                                        activation_dict[args.activation](),
                                        torch.nn.Dropout(p=args.dropout),
                                        torch.nn.Linear(10, 1))

        eps = args.lr
        criterion_dict = {'mse': torch.nn.MSELoss, 'L1': torch.nn.L1Loss}
        optim = torch.optim.SGD(params=model.parameters(), lr=eps)

        # training loop neural nets
        train_loop(X_train_scaled, y_train, X_test_scaled, y_test, model, criterion_dict[args.criterion](),
                   args.criterion, eps,
                   batch_size,
                   writer,
                   optim=optim,
                   handmade=False, w=None, b=None)
