import matplotlib.pyplot as plt
import numpy as np
import torch


def f(x):
    return torch.sin(x)


if __name__ == "__main__":
    x = torch.arange(-2 * np.pi, 2 * np.pi, 0.1)
    x.requires_grad_(True)
    y = f(x)
    y.backward(torch.ones([126]))
    plt.plot(
        x.detach().numpy(),
        y.detach().numpy(),
        color="blue",
        label="f(x)=sin(x)",
    )
    plt.plot(
        x.detach().numpy(),
        x.grad.detach().numpy(),
        color="green",
        label="f'(x)",
    )
    plt.title("f(x) and its derivative", fontsize=15)
    plt.xlabel("X", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.legend()
    plt.show()
