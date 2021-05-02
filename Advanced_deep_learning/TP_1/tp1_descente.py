import torch
from torch.utils.tensorboard import SummaryWriter
from src.tp1 import MSE, Linear, Context

# Create random dataset
x = torch.randn(50, 13, dtype=torch.float64)
y = torch.randn(50, 3, dtype=torch.float64)

# Parameters of our linear model
w = torch.randn(13, 3, dtype=torch.float64)
b = torch.randn(3, dtype=torch.float64)

# learning rate
epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    mse_context = Context()
    linear_context = Context()

    # Forward pass
    yhat = Linear.forward(ctx=linear_context, x_predict=x, W=w, b=b)
    loss = MSE.forward(ctx=mse_context, yhat=yhat, decoder_input=y)

    writer.add_scalar('Loss/train', loss, n_iter)
    print(f"Iteration {n_iter}: loss {loss}")

    # Backward pass: get gradient of our loss wrt w and b (grad_w, grad_b)
    grad_yhat, _ = MSE.backward(ctx=mse_context, grad_output=1)
    _, grad_w, grad_b = Linear.backward(ctx=linear_context, grad_output=grad_yhat)

    # Update our model's weights
    w -= epsilon * grad_w
    b -= epsilon * grad_b
