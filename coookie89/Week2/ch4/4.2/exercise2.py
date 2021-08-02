# Try adding an additional hidden layer to see how it affects the results.

import torch
from torch import nn
from d2l import torch as d2l


def relu(X):
    """relu(x)=max(x,0)"""
    constant_zero = torch.zeros_like(X)
    return torch.max(X, constant_zero)


def net(X):
    X = X.reshape(-1, num_inputs)
    out = relu(torch.matmul(X, W1) + b1)
    out = relu(torch.matmul(out, W2) + b2)
    return torch.matmul(out, W3) + b3


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784  # 28*28 pixels
    num_outputs = 10  # 分成10個類別
    num_hiddens_1 = 256
    num_hiddens_2 = 128

    W1 = nn.Parameter(
        torch.randn(num_inputs, num_hiddens_1, requires_grad=True) * 0.01
    )
    b1 = nn.Parameter(torch.zeros(num_hiddens_1, requires_grad=True))
    W2 = nn.Parameter(
        torch.randn(num_hiddens_1, num_hiddens_2, requires_grad=True) * 0.01
    )
    b2 = nn.Parameter(torch.zeros(num_hiddens_2, requires_grad=True))
    W3 = nn.Parameter(
        torch.randn(num_hiddens_2, num_outputs, requires_grad=True) * 0.01
    )
    b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2, W3, b3]

    loss = nn.CrossEntropyLoss()

    num_epochs = 10
    lr = 0.1  # learning rate

    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    d2l.predict_ch3(net, test_iter)
