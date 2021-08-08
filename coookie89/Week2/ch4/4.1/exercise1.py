# Compute the derivative of the pReLU activation function.

import torch
from d2l import torch as d2l


def PReLU(x, a):
    """define PReLU(x)=max(x,0)+a*min(x,0)"""
    constant_zero = torch.FloatTensor([0.0])
    return torch.max(x, constant_zero) + (a * torch.min(x, constant_zero))


if __name__ == "__main__":
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = PReLU(x, 0.25)

    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, "x", "grad of relu", figsize=(5, 2.5))
