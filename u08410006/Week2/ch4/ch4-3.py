import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    """
    initial the weights
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":
    net = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)
    )

    net.apply(init_weights)

    BATCH_SIZE, LR, NUM_EPOCHS = 256, 0.1, 10
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=LR)

    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)
    d2l.train_ch3(net, train_iter, test_iter, loss, NUM_EPOCHS, trainer)
