# Show that tanh(x)+1=2*sigmoid(2x).

import torch
from d2l import torch as d2l


def question_function_1(x):
    return torch.tanh(x) + 1


def question_function_2(x):
    return 2 * torch.sigmoid(2 * x)


if __name__ == "__main__":
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

    y1 = question_function_1(x)
    d2l.plot(x.detach(), y1.detach(), "x", "tanh(x)+1", figsize=(5, 2.5))

    y2 = question_function_2(x)
    d2l.plot(x.detach(), y2.detach(), "x", "2*sigmoid(2x)", figsize=(5, 2.5))

    print("y1==y2 ==>\n", y1.detach() == y2.detach(), "\n")
