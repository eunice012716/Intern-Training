# 5.2. Parameter Management

import torch
from torch import nn


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f"block {i}", block1())
    return net


def init_normal(m):
    """
    initializes all weight parameters as Gaussian random variables with standard deviation 0.01,
    while bias parameters cleared to zero.
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    """initialize all weight parameters to a given constant value (say, 1)"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in m.named_parameters()][0],
        )
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


if __name__ == "__main__":

    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    net(X)

    # Parameter Access
    print("Parameter Access------------------------------------------------")

    print("net[2].state_dict() ==>\n", net[2].state_dict(), "\n")
    print("type(net[2].bias) ==>\n", type(net[2].bias), "\n")
    print("net[2].bias ==>\n", net[2].bias, "\n")
    print("net[2].bias.data ==>\n", net[2].bias.data, "\n")
    print("net[2].weight.grad == None ==>\n", net[2].weight.grad is None, "\n")

    # All Parameters at Once
    print("All Parameters at Once------------------------------------------")

    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    print(net.state_dict()["2.bias"].data)
    print()

    # Collecting Parameters from Nested Blocks
    print("Collecting Parameters from Nested Blocks------------------------")

    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print("rgnet(X) ==>\n", rgnet(X), "\n")
    print("rgnet ==>\n", rgnet, "\n")
    print("rgnet[0][1][0].bias.data ==>\n", rgnet[0][1][0].bias.data, "\n")

    # Parameter Initialization
    print("Parameter Initialization----------------------------------------")

    net.apply(init_normal)
    print("initial parameter with normal")
    print("net[0].weight.data[0] ==>\n", net[0].weight.data[0])
    print("net[0].bias.data[0] ==>\n", net[0].bias.data[0], "\n")

    net.apply(init_constant)
    print("initial parameter with constant(1)")
    print("net[0].weight.data[0] ==>\n", net[0].weight.data[0])
    print("net[0].bias.data[0] ==>\n", net[0].bias.data[0], "\n")

    print("apply different initializers for certain blocks")
    net[0].apply(
        xavier
    )  # initialize the first layer with the Xavier initializer
    net[2].apply(
        init_42
    )  # initialize the second layer to a constant value of 42
    print("net[0].weight.data[0] ==>\n", net[0].weight.data[0])
    print("net[2].weight.data ==>\n", net[2].weight.data, "\n")

    # Custom Initialization
    print("Custom Initialization-------------------------------------------")

    net.apply(my_init)
    print("net[0].weight[:2] ==>\n", net[0].weight[:2], "\n")
    net[0].weight.data[:] += 1
    net[0].weight.data[0, 0] = 42
    print("net[0].weight.data[0] ==>\n", net[0].weight.data[0], "\n")

    # Tied Parameters
    print("Tied Parameters-------------------------------------------------")

    # We need to give the shared layer a name so that we can refer to its parameters
    shared = nn.Linear(8, 8)
    net = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        shared,
        nn.ReLU(),
        shared,
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    net(X)
    # Check whether the parameters are the same
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    # Make sure that they are actually the same object rather than just having the
    # same value
    print(net[2].weight.data[0] == net[4].weight.data[0])
