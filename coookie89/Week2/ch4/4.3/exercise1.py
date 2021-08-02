# Try adding different numbers of hidden layers (you may also modify the learning rate). What setting works best?

import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":

    # 兩層hidden layers
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    net.apply(init_weights)

    batch_size = 256
    lr = 0.5
    num_epochs = 10

    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    print("Train Accuracy ==> \n", d2l.evaluate_accuracy(net, train_iter), "\n")
    print("Test Accuracy ==> \n", d2l.evaluate_accuracy(net, test_iter), "\n")
