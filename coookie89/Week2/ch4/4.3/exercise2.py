# Try out different activation functions. Which one works best?

import torch
from torch import nn
from d2l import torch as d2l


BATCH_SIZE = 256
NUM_EPOCHS, LR = 10, 0.3


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def apply_activation_functions(activation_function):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        activation_function,
        nn.Linear(256, 10),
    )


if __name__ == "__main__":
    nets = []
    nets.append(apply_activation_functions(nn.ReLU()))
    nets.append(apply_activation_functions(nn.Sigmoid()))
    nets.append(apply_activation_functions(nn.Tanh()))

    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)

    for net_id in range(0, 3):
        if net_id == 0:
            print("using ReLU function\n")
        elif net_id == 1:
            print("using Sigmoid function\n")
        elif net_id == 2:
            print("using Tanh function\n")
        else:
            print(f"{net_id=}")

        nets[net_id].apply(init_weights)

        loss = nn.CrossEntropyLoss()
        trainer = torch.optim.SGD(nets[net_id].parameters(), lr=LR)

        d2l.train_ch3(
            nets[net_id], train_iter, test_iter, loss, NUM_EPOCHS, trainer
        )

    print("using ReLU() as activation functions is better.")
