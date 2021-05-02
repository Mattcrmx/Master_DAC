import torch
from torch.autograd import Function


class Context:
    """ very simple context object
     used to mimic Pytorch
    """

    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *args):
        self._saved_tensors = args

    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """MSE Loss"""

    @staticmethod
    def forward(ctx, y_hat, y):
        # keep the values for the backward pass
        ctx.save_for_backward(y_hat, y)

        # shape of y_hat and y: (q, p)
        q, p = y_hat.shape
        assert y_hat.shape == y.shape

        output = (1 / q) * ((y_hat - y) @ (y_hat - y).T).trace()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Computes the gradient w.r.t each input
        y_hat, y = ctx.saved_tensors

        # shape of y_hat and y: (q, p)
        q, p = y_hat.shape
        assert y_hat.shape == y.shape

        grad_y_hat = grad_output * 2 / q * (y_hat - y)
        grad_y = grad_output * -2 / q * (y_hat - y)
        return grad_y_hat, grad_y


class Linear(Function):
    """Linear regression"""

    @staticmethod
    def forward(ctx, x, w, b):
        # keep the values for the backward pass
        ctx.save_for_backward(x, w, b)

        q, n = x.shape
        p, = b.shape
        assert (n, p) == w.shape

        # using PyTorch broadcast, b is added to the last dimension of X @ W (the rows)
        output = x @ w + b
        # output shape: (q, p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Computes the gradient w.r.t each input
        x, w, b = ctx.saved_tensors

        # shape of
        # X: (q, n)
        # W: (n, p)
        # b: (p,)
        # grad_output (d(output_next_layer)/dY): (q, p)
        q, n = x.shape
        p, = b.shape
        assert (q, p) == grad_output.shape

        grad_x = grad_output @ w.T
        grad_w = x.T @ grad_output
        grad_b = torch.ones(q, dtype=torch.float32) @ grad_output
        return grad_x, grad_w, grad_b


def train_loop(x_train_scaled, y_train, x_test_scaled, y_test, model, criterion, loss_type, eps, batch_size, writer,
               optim=None,
               handmade=True, w=None, b=None):
    for epoch in range(100):

        permutation = torch.randperm(x_train_scaled.size()[0])
        epoch_loss = []

        for i in range(0, x_train_scaled.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train_scaled[indices], y_train[indices]

            if handmade:
                # using the linear regression previously coded
                y_hat = model(batch_x, w, b)
                loss = criterion(y_hat, batch_y)

                loss.backward()
                epoch_loss.append(loss.item())

                with torch.no_grad():
                    w -= eps * w.grad
                    b -= eps * b.grad

                w.grad.data.zero_()
                b.grad.data.zero_()

                with torch.no_grad():
                    y_hat_test = Linear.apply(x_test_scaled, w, b)
                    test_loss = MSE.apply(y_hat_test, y_test)

            else:
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss.append(loss.item())

                with torch.no_grad():
                    y_hat_test = model(x_test_scaled)
                    test_loss = criterion(y_hat_test, y_test)

        train_loss = sum(epoch_loss) / len(epoch_loss)
        print(train_loss)
        writer.add_scalar(f"{loss_type}/train", train_loss, epoch)
        writer.add_scalar(f"{loss_type}/test", test_loss, epoch)


mse = MSE.apply
linear = Linear.apply
