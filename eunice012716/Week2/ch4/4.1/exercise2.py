import torch
from d2l import torch as d2l


def modified_ReLU(a: int, b: int, c: int, x: torch.Tensor):
    num = x.shape[0]
    modified_x = torch.zeros(num)
    for i in range(0, num):
        modified_x[i] = max(0, a * (x[i] + c) + b)

    return modified_x


def aggregating_ReLU(a: int, b: int, c: int, d: int, x: torch.Tensor):
    num = x.shape[0]
    aggregating_x = torch.zeros(num)
    for i in range(0, num):
        aggregating_x[i] = max(0, c * (x[i] - a)) - max(0, d * (x[i] - b))

    return aggregating_x


if __name__ == "__main__":
    x = torch.arange(-8.0, 8.0, 0.1)
    y = torch.relu(x)
    m = modified_ReLU(2, 4, 1, x)
    a = aggregating_ReLU(1, 2, 3, 4, x)
    piecewise_sum = m + a
    d2l.plot(
        x.detach(),
        piecewise_sum.detach(),
        "x",
        "piecewise function",
        figsize=(5, 2.5),
    )
