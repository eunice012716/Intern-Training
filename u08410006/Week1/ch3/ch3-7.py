import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_EPOCHS = 10

    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)

    # PyTorch does not implicitly reshape the inputs. Thus we define the flatten
    # layer to reshape the inputs before the linear layer in our network
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    d2l.train_ch3(net, train_iter, test_iter, loss, NUM_EPOCHS, trainer)
