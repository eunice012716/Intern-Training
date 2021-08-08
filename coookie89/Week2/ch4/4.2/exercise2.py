# Try adding an additional hidden layer to see how it affects the results.

import torch
from torch import nn
from d2l import torch as d2l


BATCH_SIZE = 256
NUM_INPUTS, NUM_OUTPUTS = (
    784,
    10,
)  # NUM_INPUTS: 28*28 =784 pixels, NUM_OUTPUTS: 分成10個類別
NUM_HIDDENS_1, NUM_HIDDENS_2 = 256, 128
NUM_EPOCHS, LR = 10, 0.1


def relu(X):
    """relu(x)=max(x,0)"""
    constant_zero = torch.zeros_like(X)
    return torch.max(X, constant_zero)


def net(X):
    X = X.reshape(-1, NUM_INPUTS)
    out = relu(torch.matmul(X, W1) + b1)
    out = relu(torch.matmul(out, W2) + b2)
    return torch.matmul(out, W3) + b3


if __name__ == "__main__":
    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)
    scale = 0.01  # to scale the parameters to this range

    W1 = nn.Parameter(
        torch.randn(NUM_INPUTS, NUM_HIDDENS_1, requires_grad=True) * scale
    )
    b1 = nn.Parameter(torch.zeros(NUM_HIDDENS_1, requires_grad=True))
    W2 = nn.Parameter(
        torch.randn(NUM_HIDDENS_1, NUM_HIDDENS_2, requires_grad=True) * scale
    )
    b2 = nn.Parameter(torch.zeros(NUM_HIDDENS_2, requires_grad=True))
    W3 = nn.Parameter(
        torch.randn(NUM_HIDDENS_2, NUM_OUTPUTS, requires_grad=True) * scale
    )
    b3 = nn.Parameter(torch.zeros(NUM_OUTPUTS, requires_grad=True))

    params = [W1, b1, W2, b2, W3, b3]

    loss_function = nn.CrossEntropyLoss()

    updater = torch.optim.SGD(params, lr=LR)
    d2l.train_ch3(
        net, train_iter, test_iter, loss_function, NUM_EPOCHS, updater
    )

    d2l.predict_ch3(net, test_iter)
