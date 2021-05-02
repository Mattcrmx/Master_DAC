import torch


class MyNetwork(torch.nn.Module):
    def __init__(self, d_in, h, d_out):
        super(MyNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(d_in, h)
        self.tan_h = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(h, d_out)

    def forward(self, X):
        y_hat = self.tan_h(self.linear1(X))
        y_hat = self.linear2(y_hat)

        return y_hat

