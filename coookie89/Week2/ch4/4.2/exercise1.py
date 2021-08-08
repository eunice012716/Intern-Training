# Change the value of the hyperparameter num_hiddens and see how this hyperparameter influences your results.
# Determine the best value of this hyperparameter, keeping all others constant.

import torch
from torch import nn
from d2l import torch as d2l

BATCH_SIZE = 256
NUM_INPUTS, NUM_OUTPUTS = (
    784,
    10,
)  # NUM_INPUTS: 28*28 =784 pixels, NUM_OUTPUTS: 分成10個類別
NUM_HIDDENS = [1024, 512, 256, 128]
NUM_EPOCHS, LR = 10, 0.1


def relu(X):
    """relu(x)=max(x,0)"""
    constant_zero = torch.zeros_like(X)
    return torch.max(X, constant_zero)


def net(X):
    X = X.reshape((-1, NUM_INPUTS))
    H = relu(X @ W1 + b1)  # Here '@' stands for matrix multiplication
    return H @ W2 + b2


if __name__ == "__main__":
    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)
    scale = 0.01  # to scale the parameters to this range

    for num_hidden_test in NUM_HIDDENS:
        print("hidden layer node num = ", num_hidden_test, "\n")

        W1 = nn.Parameter(
            torch.randn(NUM_INPUTS, num_hidden_test, requires_grad=True) * scale
        )
        b1 = nn.Parameter(torch.zeros(num_hidden_test, requires_grad=True))
        W2 = nn.Parameter(
            torch.randn(num_hidden_test, NUM_OUTPUTS, requires_grad=True)
            * scale
        )
        b2 = nn.Parameter(torch.zeros(NUM_OUTPUTS, requires_grad=True))

        params = [W1, b1, W2, b2]

        updater = torch.optim.SGD(params, lr=LR)
        d2l.train_ch3(
            net,
            train_iter,
            test_iter,
            nn.CrossEntropyLoss(),
            NUM_EPOCHS,
            updater,
        )

        d2l.predict_ch3(net, test_iter)
