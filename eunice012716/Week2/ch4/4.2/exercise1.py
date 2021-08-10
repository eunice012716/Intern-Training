import torch
from torch import nn
from d2l import torch as d2l

BATCH_SIZE = 256
NUM_INPUTS = 784


def relu(X):
    """relu(x)=max(x,0)"""
    constant_zero = torch.zeros_like(X)
    return torch.max(X, constant_zero)


def net(X, num_inputs=NUM_INPUTS):
    X = X.reshape((-1, num_inputs))
    Hidden_1 = relu(X @ W1 + b1)
    return Hidden_1 @ W2 + b2


if __name__ == "__main__":
    NUM_OUTPUTS = 10
    NUM_HIDDENS = [64, 128, 256, 512, 1024, 2048]
    NUM_EPOCHS, LR = 10, 0.1
    TRAIN_ITER, TEST_ITER = d2l.load_data_fashion_mnist(BATCH_SIZE)
    for num_hiddens in NUM_HIDDENS:
        W1 = nn.Parameter(
            torch.randn(NUM_INPUTS, num_hiddens, requires_grad=True) * 0.01
        )
        b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
        W2 = nn.Parameter(
            torch.randn(num_hiddens, NUM_OUTPUTS, requires_grad=True) * 0.01
        )
        b2 = nn.Parameter(torch.zeros(NUM_OUTPUTS, requires_grad=True))

        params = [W1, b1, W2, b2]

        updater = torch.optim.SGD(params, lr=LR)
        d2l.train_ch3(
            net,
            TRAIN_ITER,
            TEST_ITER,
            nn.CrossEntropyLoss(),
            NUM_EPOCHS,
            updater,
        )

        d2l.predict_ch3(net, TEST_ITER)
