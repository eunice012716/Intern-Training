# Change the value of the hyperparameter num_hiddens and see how this hyperparameter influences your results.
# Determine the best value of this hyperparameter, keeping all others constant.

import torch
from torch import nn
from d2l import torch as d2l


def relu(X):
    """relu(x)=max(x,0)"""
    constant_zero = torch.zeros_like(X)
    return torch.max(X, constant_zero)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # Here '@' stands for matrix multiplication
    return H @ W2 + b2


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784  # 28*28 pixels
    num_outputs = 10  # 分成10個類別
    num_hiddens = [1024, 512, 256, 128]

    for num_hidden_test in num_hiddens:
        print("hidden layer node num = ", num_hidden_test, "\n")

        W1 = nn.Parameter(
            torch.randn(num_inputs, num_hidden_test, requires_grad=True) * 0.01
        )
        b1 = nn.Parameter(torch.zeros(num_hidden_test, requires_grad=True))
        W2 = nn.Parameter(
            torch.randn(num_hidden_test, num_outputs, requires_grad=True) * 0.01
        )
        b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

        params = [W1, b1, W2, b2]

        loss = nn.CrossEntropyLoss()

        num_epochs = 10
        lr = 0.1  # learning rate

        updater = torch.optim.SGD(params, lr=lr)
        d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

        d2l.predict_ch3(net, test_iter)
