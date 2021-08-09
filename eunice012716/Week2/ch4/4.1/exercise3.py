import torch
from d2l import torch as d2l

if __name__ == "__main__":
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    sigmoid = 2 * torch.sigmoid(2 * x)
    tanh = torch.tanh(x) + 1
    d2l.plot(x.detach(), tanh.detach(), "x", "tanh(x) + 1", figsize=(5, 2.5))
    d2l.plot(
        x.detach(), sigmoid.detach(), "x", "2 * sigmoid(2x)", figsize=(5, 2.5)
    )
