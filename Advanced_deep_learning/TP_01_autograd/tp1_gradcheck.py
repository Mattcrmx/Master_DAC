import torch
from src.tp1 import mse, linear

if __name__ == '__main__':
    # Test du gradient de MSE et de Linear
    q, n, p = 15, 7, 5
    yhat = torch.randn(q, p, requires_grad=True, dtype=torch.float64)
    y = torch.randn(q, p, requires_grad=True, dtype=torch.float64)
    X = torch.randn(q, n, requires_grad=True, dtype=torch.float64)
    W = torch.randn(n, p, requires_grad=True, dtype=torch.float64)
    b = torch.randn(p, requires_grad=True, dtype=torch.float64)

    print(f'Gradcheck for MSE - success : {torch.autograd.gradcheck(mse, (yhat, y))}')
    print(f'Gradcheck for Linear - success : {torch.autograd.gradcheck(linear, (X, W, b))}')
