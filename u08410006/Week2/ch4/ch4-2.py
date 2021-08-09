import torch
from torch import nn
from d2l import torch as d2l


def relu(X):
    """
    activation ourselves using the maximum function
    """
    a = torch.zeros_like(X)
    return torch.max(X, a)


def build_net(X):
    """
    reshape and implement our model
    """
    X = X.reshape((-1, NUM_INPUTS))
    H = relu(X @ W1 + b1)  # Here '@' stands for matrix multiplication
    return H @ W2 + b2


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDENS = 784, 10, 256
    NUM_EPOCHS, LR = 10, 0.1
    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)

    W1 = nn.Parameter(
        torch.randn(NUM_INPUTS, NUM_HIDDENS, requires_grad=True) * 0.01
    )
    b1 = nn.Parameter(torch.zeros(NUM_HIDDENS, requires_grad=True))
    W2 = nn.Parameter(
        torch.randn(NUM_HIDDENS, NUM_OUTPUTS, requires_grad=True) * 0.01
    )
    b2 = nn.Parameter(torch.zeros(NUM_OUTPUTS, requires_grad=True))

    params = [W1, b1, W2, b2]

    updater = torch.optim.SGD(params, lr=LR)
    d2l.train_ch3(
        build_net,
        train_iter,
        test_iter,
        nn.CrossEntropyLoss(),
        NUM_EPOCHS,
        updater,
    )

    d2l.predict_ch3(build_net, test_iter)
