import torch
from torch.autograd import Function


class Context:
    """
    very simple context object
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
        grad_b = torch.ones(q, dtype=torch.float64) @ grad_output
        return grad_x, grad_w, grad_b


mse = MSE.apply
linear = Linear.apply
