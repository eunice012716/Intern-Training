import torch
from torch import nn


def block1():
    """
    produces block1 and then combine these inside yet larger blocks
    """
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    """
    produces block1 and then combine these inside yet larger blocks
    """
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f"block {i}", block1())
    return net


def init_normal(m):
    """
    initializes all weight parameters as Gaussian random variables
    with standard deviation 0.01, while bias parameters cleared to zero.
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    """
    initialize all the parameters to a given constant value
    """
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def xavier(m):
    """
    initialize the first layer with the Xavier initializer
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    """
    initialize the second layer to a constant value of 42.
    """
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


def my_init(m):
    """
    implement a my_init function to apply to net.
    """
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
    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    net.state_dict()["2.bias"].data
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    rgnet(X)
    print(rgnet)
    rgnet[0][1][0].bias.data
    net.apply(init_normal)
    net[0].weight.data[0], net[0].bias.data[0]
    net.apply(init_constant)
    net[0].weight.data[0], net[0].bias.data[0]
    net[0].apply(xavier)
    net[2].apply(init_42)
    print(net[0].weight.data[0])
    print(net[2].weight.data)
    net.apply(my_init)
    net[0].weight[:2]
    net[0].weight.data[:] += 1
    net[0].weight.data[0, 0] = 42
    net[0].weight.data[0]
    # We need to give the shared layer a name so that we can refer to its
    # parameters
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
